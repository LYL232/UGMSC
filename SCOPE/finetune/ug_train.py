import argparse
from functools import partial
import os
import torch
import math
import json

from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch import nn
from torch.utils.data.dataloader import DataLoader

from transformers.models.bert.modeling_bert import \
    BertLMPredictionHead, BertPredictionHeadTransform, BertEncoder, BertPooler, BertPreTrainedModel, \
    BertModel
from transformers import AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput
from datasets.utils import Pinyin
from datasets.bert_csc_dataset import TestCSCDataset, Dynaimic_CSCDataset
from datasets.collate_functions import collate_to_max_length_for_train_dynamic_pron_loss

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from models.glyph_embedding import GlyphEmbedding
from models.pinyin_embedding import PinyinEmbedding
from copy import deepcopy
from utils.random_seed import set_random_seed

set_random_seed(2334)


def decode_sentence_and_get_pinyinids(ids):
    dataset = TestCSCDataset(
        data_path='data/test.sighan15.pkl',
        chinese_bert_path='UGFPT',
    )
    sent = ''.join(dataset.tokenizer.decode(ids).split(' '))
    tokenizer_output = dataset.tokenizer.encode(sent)
    pinyin_tokens = dataset.convert_sentence_to_pinyin_ids(sent, tokenizer_output)
    pinyin_ids = torch.LongTensor(pinyin_tokens).unsqueeze(0)
    return sent, pinyin_ids


class MultiModalEncoder(nn.Module):
    def __init__(self, config):
        super(MultiModalEncoder, self).__init__()
        config_path = os.path.join(config.name_or_path, 'config')
        font_files = []
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pinyin_embeddings = PinyinEmbedding(
            embedding_size=128, pinyin_out_dim=config.hidden_size,
            config_path=config_path
        )
        self.glyph_embeddings = GlyphEmbedding(font_npy_files=font_files)
        self.glyph_map = nn.Linear(1728, config.hidden_size)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, pinyin_ids, position_ids=None, token_type_ids=None):
        input_shape = input_ids.size()
        inputs_embeds = self.word_embeddings(input_ids)

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        position_embeddings = self.position_embeddings(position_ids)
        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_map(self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]

        return word_embeddings, pinyin_embeddings, glyph_embeddings, position_embeddings, token_type_embeddings


class UGFusionBertEmbeddings(nn.Module):
    def __init__(self, config, merge_modal: int):
        super(UGFusionBertEmbeddings, self).__init__()
        config_path = os.path.join(config.name_or_path, 'config')
        font_files = []
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(os.path.join(config_path, file))
        self.merge_modal = merge_modal
        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.map_fc = nn.Linear(config.hidden_size * merge_modal, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_embeds,
            position_embeddings,
            token_type_embeddings
    ):
        # fusion layer
        input_embeds = torch.cat(input_embeds, dim=2)
        input_embeds = self.map_fc(input_embeds)

        embeddings = input_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NumericalMLP(nn.Module):
    def __init__(
            self,
            in_features: int, hidden_size, out_features: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.dense_0 = nn.Linear(in_features * 4, hidden_size)
        self.dense_1 = nn.Linear(hidden_size, hidden_size)
        self.dense_2 = nn.Linear(hidden_size, hidden_size)
        self.mapping = nn.Linear(hidden_size, out_features)
        self.dropout = nn.Dropout(dropout)
        self.hidden_act = nn.ReLU()

    def forward(self, inputs):
        # features = torch.cat(multiscale_uncertainty_features(inputs), dim=-1)  # (b, s, in_features)
        hiddens = self.dropout(self.hidden_act(self.dense_0(inputs)))
        hiddens = self.dropout(self.hidden_act(self.dense_1(hiddens)))
        hiddens = self.dropout(self.hidden_act(self.dense_2(hiddens)))
        return self.mapping(hiddens)


def multiscale_uncertainty_features(m):
    im = 1 - m
    return [m, im, torch.log(m + 1e-12), torch.log(im + 1e-12)]


class UGGlyceBertModel(BertModel):
    def __init__(self, config):
        super(UGGlyceBertModel, self).__init__(config)
        self.config = config
        self.embeddings = None
        self.encoder = None

        self.multimodal_encoder = MultiModalEncoder(config)
        self.semantic_embeddings = UGFusionBertEmbeddings(config, 1)
        self.phonetic_embeddings = UGFusionBertEmbeddings(config, 2)
        self.graphic_embeddings = UGFusionBertEmbeddings(config, 2)
        self.joint_embeddings = UGFusionBertEmbeddings(config, 3)
        corrector_config = deepcopy(config)
        corrector_config.num_hidden_layers = 3
        self.semantic_corrector = BertEncoder(corrector_config)
        self.phonetic_corrector = BertEncoder(corrector_config)
        self.graphic_corrector = BertEncoder(corrector_config)
        self.joint_corrector = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.modal_weight_network = NumericalMLP(
            in_features=6,
            hidden_size=corrector_config.hidden_size,
            out_features=3,
            dropout=corrector_config.hidden_dropout_prob
        )
        self.modal_weight_act = nn.Sigmoid()

        self.correct_weight_network = NumericalMLP(
            in_features=8,
            hidden_size=corrector_config.hidden_size,
            out_features=4,
            dropout=corrector_config.hidden_dropout_prob
        )

        self.max_correct_entropy = math.log(config.vocab_size)
        self.max_detect_entropy = math.log(2)
        self.eps = 1e-12

        self.predictions = BertLMPredictionHead(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.multimodal_encoder.word_embeddings

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            past_key_values=None,
            use_cache=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        word_embeddings, pinyin_embeddings, glyph_embeddings, \
        position_embeddings, token_type_embeddings = \
            self.multimodal_encoder(input_ids, pinyin_ids, token_type_ids=token_type_ids)

        # partial corrector
        input_ids = input_ids.view(input_shape[0], input_shape[1], 1).long()
        semantic_logits, semantic_unc = self.modality_corrector_forward(
            self.semantic_corrector(
                self.semantic_embeddings(
                    input_embeds=[word_embeddings],
                    position_embeddings=position_embeddings,
                    token_type_embeddings=token_type_embeddings
                ),
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )[0],
            input_ids
        )
        phonetic_logits, phonetic_unc = self.modality_corrector_forward(
            self.phonetic_corrector(
                self.phonetic_embeddings(
                    input_embeds=[word_embeddings, pinyin_embeddings],
                    position_embeddings=position_embeddings,
                    token_type_embeddings=token_type_embeddings
                ),
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )[0],
            input_ids
        )

        graphic_logits, graphic_unc = self.modality_corrector_forward(
            self.graphic_corrector(
                self.graphic_embeddings(
                    input_embeds=[word_embeddings, glyph_embeddings],
                    position_embeddings=position_embeddings,
                    token_type_embeddings=token_type_embeddings
                ),
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )[0],
            input_ids
        )

        uncertainty_metrics = torch.cat([
            *semantic_unc,
            *phonetic_unc,
            *graphic_unc
        ], dim=-1)
        modal_weights = self.modal_weight_act(
            self.modal_weight_network(uncertainty_metrics)
        )

        joint_outputs = self.joint_corrector(
            self.joint_embeddings(
                input_embeds=[
                    word_embeddings * modal_weights[:, :, 0:1],
                    pinyin_embeddings * modal_weights[:, :, 1:2],
                    glyph_embeddings * modal_weights[:, :, 2:3]
                ],
                position_embeddings=position_embeddings,
                token_type_embeddings=token_type_embeddings
            ),
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        joint_sequence_output = joint_outputs[0]

        joint_logits, joint_unc = self.modality_corrector_forward(
            joint_sequence_output, input_ids
        )

        weight = torch.softmax(
            self.correct_weight_network(torch.cat([
                uncertainty_metrics,
                *joint_unc
            ], dim=-1)),
            dim=-1
        )
        logits = semantic_logits * weight[:, :, 0:1] \
                 + phonetic_logits * weight[:, :, 1:2] \
                 + graphic_logits * weight[:, :, 2:3] \
                 + joint_logits * weight[:, :, 3:4]

        pooled_output = self.pooler(joint_sequence_output) if self.pooler is not None else None

        return joint_sequence_output, logits, \
               semantic_logits, phonetic_logits, graphic_logits, joint_logits, \
               pooled_output, \
               joint_outputs.hidden_states, \
               joint_outputs.attentions

    def modality_corrector_forward(self, hidden, input_ids):
        logits = self.predictions(hidden)
        prob = torch.softmax(logits.detach(), dim=-1)
        correct_entropy = -1 * torch.sum(
            prob * torch.log(prob + self.eps),
            dim=-1, keepdim=True
        ) / self.max_correct_entropy
        input_ids_prob = torch.gather(prob, dim=-1, index=input_ids)
        err_prob = 1 - input_ids_prob
        detect_entropy = -1 * (
                input_ids_prob * torch.log(input_ids_prob + self.eps) +
                err_prob * torch.log(err_prob + self.eps)
        ) / self.max_detect_entropy
        return logits, [
            *multiscale_uncertainty_features(correct_entropy),
            *multiscale_uncertainty_features(detect_entropy),
        ]


class Phonetic_Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pinyin = Pinyin()
        self.transform = BertPredictionHeadTransform(config)
        self.sm_classifier = nn.Linear(config.hidden_size, self.pinyin.sm_size)
        self.ym_classifier = nn.Linear(config.hidden_size, self.pinyin.ym_size)
        self.sd_classifier = nn.Linear(config.hidden_size, self.pinyin.sd_size)

    def forward(self, sequence_output):
        sequence_output = self.transform(sequence_output)
        sm_scores = self.sm_classifier(sequence_output)
        ym_scores = self.ym_classifier(sequence_output)
        sd_scores = self.sd_classifier(sequence_output)
        return sm_scores, ym_scores, sd_scores


class UGMultiTaskHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Phonetic_relationship = Phonetic_Classifier(config)

    def forward(self, sequence_output):
        sm_scores, ym_scores, sd_scores = self.Phonetic_relationship(sequence_output)
        return sm_scores, ym_scores, sd_scores


class UGDynamic_GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(UGDynamic_GlyceBertForMultiTask, self).__init__(config)

        self.cls = UGMultiTaskHeads(config)

        self.bert = UGGlyceBertModel(config)

        self.loss_fct = CrossEntropyLoss(reduction='none')

        self.init_weights()

    def get_output_embeddings(self):
        return self.bert.predictions.decoder

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            tgt_pinyin_ids=None,
            pinyin_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            gamma=1,
            var=1,
            **kwargs
    ):

        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        loss_mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        outputs_x = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_x = outputs_x[0]
        prediction_scores = outputs_x[1]
        if tgt_pinyin_ids is not None:
            with torch.no_grad():
                outputs_y = self.bert(
                    labels,
                    tgt_pinyin_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                encoded_y = outputs_y[0]
                pron_x = self.cls.Phonetic_relationship.transform(encoded_x)
                pron_y = self.cls.Phonetic_relationship.transform(encoded_y)  # [bs, seq, hidden_states]
                assert pron_x.shape == pron_y.shape
                sim_xy = F.cosine_similarity(pron_x, pron_y, dim=-1)  # [ns, seq]
                factor = torch.exp(-((sim_xy - 1.0) / var).pow(2)).detach()

        sm_scores, ym_scores, sd_scores = self.cls(encoded_x)

        masked_lm_loss = None
        phonetic_loss = None
        loss_fct = self.loss_fct  # -100 index = padding token
        if labels is not None and pinyin_labels is not None:
            active_loss = loss_mask.view(-1) == 1

            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), active_labels)
            # DCT training
            for each in outputs_x[2:6]:
                masked_lm_loss += loss_fct(each.view(-1, self.config.vocab_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 0].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sm_loss = loss_fct(sm_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sm_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 1].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            ym_loss = loss_fct(ym_scores.view(-1, self.cls.Phonetic_relationship.pinyin.ym_size), active_labels)

            active_labels = torch.where(
                active_loss, pinyin_labels[..., 2].view(-1), torch.tensor(loss_fct.ignore_index).type_as(pinyin_labels)
            )
            sd_loss = loss_fct(sd_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sd_size), active_labels)
            phonetic_loss = (sm_loss + ym_loss + sd_loss) / 3

            def weighted_mean(weight, input):
                return torch.sum(weight * input) / torch.sum(weight)

            masked_lm_loss = weighted_mean(torch.ones_like(masked_lm_loss), masked_lm_loss)
            phonetic_loss = weighted_mean(factor.view(-1), phonetic_loss)

        loss = None
        if masked_lm_loss is not None and phonetic_loss is not None:
            loss = masked_lm_loss / 5
            loss += phonetic_loss * gamma

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs_x[7],
            attentions=outputs_x[8],
        )


class UGCSCTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        self.bert_dir = args.bert_path
        self.bert_config = BertConfig.from_pretrained(
            self.bert_dir, output_hidden_states=False
        )
        self.model = UGDynamic_GlyceBertForMultiTask.from_pretrained(self.bert_dir)
        if args.ckpt_path is not None:
            print("loading from ", args.ckpt_path)
            ckpt = torch.load(args.ckpt_path, )["state_dict"]
            new_ckpt = {}
            for key in ckpt.keys():
                new_ckpt[key[6:]] = ckpt[key]
            self.model.load_state_dict(new_ckpt, strict=False)
            print(self.model.device, torch.cuda.is_available())
        self.vocab_size = self.bert_config.vocab_size

        self.loss_fct = CrossEntropyLoss()
        gpus_string = (
            str(self.args.gpus) if not str(self.args.gpus).endswith(",") else str(self.args.gpus)[:-1]
        )
        self.num_gpus = len(gpus_string.split(","))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),  # according to RoBERTa paper
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
        )
        t_total = (
                len(self.train_dataloader())
                // self.args.accumulate_grad_batches
                * self.args.max_epochs
        )
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_ids, labels=None, pinyin_labels=None, tgt_pinyin_ids=None, var=1):
        """"""
        attention_mask = (input_ids != 0).long()
        return self.model(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            labels=labels,
            tgt_pinyin_ids=tgt_pinyin_ids,
            pinyin_labels=pinyin_labels,
            gamma=self.args.gamma if 'gamma' in self.args else 0,
        )

    def compute_loss(self, batch):
        input_ids, pinyin_ids, labels, tgt_pinyin_ids, pinyin_labels = batch
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        tgt_pinyin_ids = tgt_pinyin_ids.view(batch_size, length, 8)
        outputs = self.forward(
            input_ids, pinyin_ids, labels=labels, pinyin_labels=pinyin_labels, tgt_pinyin_ids=tgt_pinyin_ids,
            var=self.args.var if 'var' in self.args else 1
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        """"""
        loss = self.compute_loss(batch)
        tf_board_logs = {
            "train_loss": loss.item(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        # torch.cuda.empty_cache()
        return {"loss": loss, "log": tf_board_logs}

    def validation_step(self, batch, batch_idx):
        """"""
        input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tokens_size = batch
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        logits = self.forward(
            input_ids,
            pinyin_ids,
        ).logits
        predict_scores = F.softmax(logits, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask
        return {
            "tgt_idx": labels.cpu(),
            "pred_idx": predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }

    def validation_epoch_end(self, outputs):
        from metrics.metric import Metric

        # print(len(outputs))
        metric = Metric(vocab_path=self.args.bert_path)
        pred_txt_path = os.path.join(self.args.save_path, "preds.txt")
        pred_lbl_path = os.path.join(self.args.save_path, "labels.txt")
        if len(outputs) == 2:
            self.log("df", 0)
            self.log("cf", 0)
            return {"df": 0, "cf": 0}
        results = metric.metric(
            batches=outputs,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=self.args.label_file,
        )
        self.log("df", results["sent-detect-f1"])
        self.log("cf", results["sent-correct-f1"])
        return {"df": results["sent-detect-f1"], "cf": results["sent-correct-f1"]}

    def train_dataloader(self) -> DataLoader:
        name = "train_all"

        dataset = Dynaimic_CSCDataset(
            data_path=os.path.join(self.args.data_dir, name),
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = dataset.tokenizer

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_for_train_dynamic_pron_loss, fill_values=[0, 0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def val_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan15.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        print('dev dataset', len(dataset))
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def test13_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan13.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def test14_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan14.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def test15_dataloader(self):
        dataset = TestCSCDataset(
            data_path='data/test.sighan15.pkl',
            chinese_bert_path=self.args.bert_path,
            max_length=self.args.max_length,
        )
        self.tokenizer = dataset.tokenizer
        from datasets.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, pinyin_ids, labels, pinyin_labels, ids, srcs, tokens_size = batch
        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        batch_size, length = input_ids.shape
        pinyin_ids = pinyin_ids.view(batch_size, length, 8)
        logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids).logits
        predict_scores = F.softmax(logits, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask

        if '13' in self.args.label_file:
            predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (
                    predict_labels == self.tokenizer.token_to_id('得'))] = \
                input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (
                        predict_labels == self.tokenizer.token_to_id('得'))]

        pre_predict_labels = predict_labels
        for _ in range(1):
            record_index = []
            for i, (a, b) in enumerate(zip(list(input_ids[0, 1:-1]), list(predict_labels[0, 1:-1]))):
                if a != b:
                    record_index.append(i)

            input_ids[0, 1:-1] = predict_labels[0, 1:-1]
            sent, new_pinyin_ids = decode_sentence_and_get_pinyinids(input_ids[0, 1:-1].cpu().numpy().tolist())
            if new_pinyin_ids.shape[1] == input_ids.shape[1]:
                pinyin_ids = new_pinyin_ids
            pinyin_ids = pinyin_ids.to(input_ids.device)
            # print(input_ids.device, pinyin_ids.device)
            logits = self.forward(input_ids=input_ids, pinyin_ids=pinyin_ids).logits
            predict_scores = F.softmax(logits, dim=-1)
            predict_labels = torch.argmax(predict_scores, dim=-1) * mask

            for i, (a, b) in enumerate(zip(list(input_ids[0, 1:-1]), list(predict_labels[0, 1:-1]))):
                if a != b and any([abs(i - x) <= 1 for x in record_index]):
                    print(ids, srcs)
                    print(i + 1, )
                else:
                    predict_labels[0, i + 1] = input_ids[0, i + 1]
            if predict_labels[0, i + 1] == input_ids[0, i + 1]:
                break
            if '13' in self.args.label_file:
                predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (
                        predict_labels == self.tokenizer.token_to_id('得'))] = \
                    input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (
                            predict_labels == self.tokenizer.token_to_id('得'))]
        # if not pre_predict_labels.equal(predict_labels):
        #     print([self.tokenizer.id_to_token(id) for id in pre_predict_labels[0][1:-1]])
        #     print([self.tokenizer.id_to_token(id) for id in predict_labels[0][1:-1]])
        return {
            "tgt_idx": labels.cpu(),
            "post_pred_idx": predict_labels.cpu(),
            "pred_idx": pre_predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--bert_path", required=True, type=str, help="bert config file")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument(
        "--label_file",
        default="data/test.sighan15.lbl.tsv",
        type=str,
        help="label file",
    )
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument(
        "--workers", type=int, default=8, help="num workers for dataloader"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument(
        "--use_memory",
        action="store_true",
        help="load datasets to memory to accelerate.",
    )
    parser.add_argument(
        "--max_length", default=512, type=int, help="max length of datasets"
    )
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument(
        "--save_topk", default=5, type=int, help="save topk checkpoint"
    )
    parser.add_argument("--mode", default="train", type=str, help="train or evaluate")
    parser.add_argument(
        "--warmup_proporation", default=0.01, type=float, help="warmup proporation"
    )
    parser.add_argument("--gamma", default=1, type=float, help="phonetic loss weight")
    parser.add_argument(
        "--ckpt_path", default=None, type=str, help="resume_from_checkpoint"
    )
    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = UGCSCTask(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "checkpoint"),
        filename="{epoch}-{df:.4f}-{cf:.4f}",
        save_top_k=args.save_topk,
        monitor="cf",
        mode="max",
    )
    logger = TensorBoardLogger(save_dir=args.save_path, name="log")

    # save args
    if not os.path.exists(os.path.join(args.save_path, "checkpoint")):
        os.mkdir(os.path.join(args.save_path, "checkpoint"))
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    trainer = Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger
    )

    trainer.fit(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()

