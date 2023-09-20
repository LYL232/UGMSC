import math
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from copy import deepcopy
import opencc
import numpy as np
import os
from PIL import ImageFont
import torch
from src.utils import pho2_convertor
from src.models import CharResNet, CharResNet1, _is_chinese_char


def multiscale_uncertainty_features(*metrics):
    features = []
    for m in metrics:
        im = 1 - m
        log_m = torch.log(m + 1e-12)
        log_im = torch.log(im + 1e-12)
        features.extend([m, im, log_m, log_im])
    return features


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

    def forward(self, *inputs):
        features = torch.cat(multiscale_uncertainty_features(*inputs), dim=-1)  # (b, s, in_features)
        hiddens = self.dropout(self.hidden_act(self.dense_0(features)))
        hiddens = self.dropout(self.hidden_act(self.dense_1(hiddens)))
        hiddens = self.dropout(self.hidden_act(self.dense_2(hiddens)))
        return self.mapping(hiddens)


class PositiveScale(nn.Module):
    def __init__(self, features: int, init_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features) * math.log(math.pow(math.e, init_scale) - 1))
        self.weight_act = nn.Softplus()

    def forward(self, inputs):
        return inputs * self.weight_act(self.weight)


class UGReaLiSe(BertPreTrainedModel):
    def __init__(
            self,
            config,
            modal_weighting_ablation: bool = False,
            ensemble_weighting_ablation: bool = False,
    ):
        super().__init__(config)
        self.config = config

        self.modal_weighting_ablation = modal_weighting_ablation
        self.ensemble_weighting_ablation = ensemble_weighting_ablation

        self.vocab_size = config.vocab_size
        self.bert = BertModel(config)

        self.pho_embeddings = nn.Embedding(pho2_convertor.get_pho_size(), config.hidden_size, padding_idx=0)
        self.pho_gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        pho_config = deepcopy(config)
        pho_config.num_hidden_layers = 4
        self.pho_model = BertModel(pho_config)

        if self.config.num_fonts == 1:
            self.char_images = nn.Embedding(config.vocab_size, 1024)
            self.char_images.weight.requires_grad = False
        else:
            self.char_images_multifonts = torch.nn.Parameter(torch.rand(21128, self.config.num_fonts, 32, 32))
            self.char_images_multifonts.requires_grad = False

        if config.image_model_type == 0:
            self.resnet = CharResNet(in_channels=self.config.num_fonts)
        elif config.image_model_type == 1:
            self.resnet = CharResNet1()
        else:
            raise NotImplementedError('invalid image_model_type %d' % config.image_model_type)
        self.resnet_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        block_config = deepcopy(config)
        block_config.num_hidden_layers = 1
        block_config.max_position_embeddings = 1
        self.semantic_blocks = BertModel(block_config)
        self.phonetic_blocks = BertModel(block_config)
        block_config = deepcopy(config)
        block_config.num_hidden_layers = 2
        block_config.max_position_embeddings = 1
        self.graphic_blocks = BertModel(block_config)

        block_config = deepcopy(config)
        block_config.num_hidden_layers = 3
        block_config.max_position_embeddings = 1
        self.joint_blocks = BertModel(block_config)

        self.semantic_blocks.embeddings.word_embeddings = None
        self.phonetic_blocks.embeddings.word_embeddings = None
        self.graphic_blocks.embeddings.word_embeddings = None
        self.joint_blocks.embeddings.word_embeddings = None
        self.pho_model.embeddings.word_embeddings = None

        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        if self.modal_weighting_ablation:
            self.modal_weight_network = None
            self.modal_scale = None
        else:
            self.modal_weight_network = NumericalMLP(
                in_features=6,
                hidden_size=config.hidden_size,
                out_features=3,
                dropout=config.hidden_dropout_prob
            )
            self.modal_scale = PositiveScale(3, 2.0)

        self.modal_weight_act = nn.Sigmoid()

        if self.ensemble_weighting_ablation:
            self.correct_weight_network = None
        else:
            self.correct_weight_network = NumericalMLP(
                in_features=8,
                hidden_size=config.hidden_size,
                out_features=4,
                dropout=config.hidden_dropout_prob
            )

        self.max_correct_entropy = math.log(self.vocab_size)
        self.max_detect_entropy = math.log(2)
        self.eps = 1e-12

        self.init_weights()

        self.evidence_act = nn.Softplus()

    def tie_cls_weight(self):
        self.classifier.weight = self.bert.embeddings.word_embeddings.weight

    def build_glyce_embed(self, vocab_dir, font_path, font_size=32):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [s.strip() for s in f]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) != 1 or (not _is_chinese_char(ord(char))):
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).reshape(char_images.shape[0], -1)
        assert char_images.shape == (21128, 1024)
        self.char_images.weight.data.copy_(char_images)

    def build_glyce_embed_multifonts(self, vocab_dir, num_fonts, use_traditional_font, font_size=32):
        font_paths = [
            ('simhei.ttf', False),
            ('xiaozhuan.ttf', False),
            ('simhei.ttf', True),
        ]
        font_paths = font_paths[:num_fonts]
        if use_traditional_font:
            font_paths = font_paths[:-1]
            font_paths.append(('simhei.ttf', True))
            self.converter = opencc.OpenCC('s2t.json')

        images_list = []
        for font_path, use_traditional in font_paths:
            images = self.build_glyce_embed_onefont(
                vocab_dir=vocab_dir,
                font_path=font_path,
                font_size=font_size,
                use_traditional=use_traditional,
            )
            images_list.append(images)

        char_images = torch.stack(images_list, dim=1).contiguous()
        self.char_images_multifonts.data.copy_(char_images)

    def build_glyce_embed_onefont(self, vocab_dir, font_path, font_size, use_traditional):
        vocab_path = os.path.join(vocab_dir, 'vocab.txt')
        with open(vocab_path) as f:
            vocab = [s.strip() for s in f.readlines()]
        if use_traditional:
            vocab = [self.converter.convert(c) if len(c) == 1 else c for c in vocab]

        font = ImageFont.truetype(font_path, size=font_size)

        char_images = []
        for char in vocab:
            if len(char) > 1:
                char_images.append(np.zeros((font_size, font_size)).astype(np.float32))
                continue
            image = font.getmask(char)
            image = np.asarray(image).astype(np.float32).reshape(image.size[::-1])  # Must be [::-1]

            # Crop
            image = image[:font_size, :font_size]

            # Pad
            if image.size != (font_size, font_size):
                back_image = np.zeros((font_size, font_size)).astype(np.float32)
                offset0 = (font_size - image.shape[0]) // 2
                offset1 = (font_size - image.shape[1]) // 2
                back_image[offset0:offset0 + image.shape[0], offset1:offset1 + image.shape[1]] = image
                image = back_image

            char_images.append(image)
        char_images = np.array(char_images)
        char_images = (char_images - np.mean(char_images)) / np.std(char_images)
        char_images = torch.from_numpy(char_images).contiguous()
        return char_images

    @staticmethod
    def build_batch(batch, tokenizer):
        src_idx = batch['src_idx'].flatten().tolist()
        chars = tokenizer.convert_ids_to_tokens(src_idx)
        pho_idx, pho_lens = pho2_convertor.convert(chars)
        batch['pho_idx'] = pho_idx
        batch['pho_lens'] = pho_lens
        return batch

    def forward(self, batch):
        input_ids = batch['src_idx']
        attention_mask = batch['masks']
        pho_idx = batch['pho_idx']
        pho_lens = batch['pho_lens']

        input_shape = input_ids.size()

        bert_hiddens = self.bert(input_ids, attention_mask=attention_mask)[0]

        pho_embeddings = self.pho_embeddings(pho_idx)
        pho_embeddings = torch.nn.utils.rnn.pack_padded_sequence(
            input=pho_embeddings,
            lengths=pho_lens,
            batch_first=True,
            enforce_sorted=False,
        )
        _, pho_hiddens = self.pho_gru(pho_embeddings)
        pho_hiddens = pho_hiddens.squeeze(0).reshape(input_shape[0], input_shape[1], -1).contiguous()
        pho_hiddens = self.pho_model(inputs_embeds=pho_hiddens, attention_mask=attention_mask)[0]

        src_idxs = input_ids.view(-1)

        if self.config.num_fonts == 1:
            images = self.char_images(src_idxs).reshape(src_idxs.shape[0], 1, 32, 32).contiguous()
        else:
            images = self.char_images_multifonts.index_select(dim=0, index=src_idxs)

        res_hiddens = self.resnet(images)
        res_hiddens = res_hiddens.reshape(input_shape[0], input_shape[1], -1).contiguous()
        res_hiddens = self.resnet_layernorm(res_hiddens)

        zero_position_ids = torch.zeros(
            input_ids.size(), dtype=torch.long,
            device=input_ids.device
        )

        uncertainty_metrics = []

        input_ids = input_ids.view(input_shape[0], input_shape[1], 1)

        semantic_logits, semantic_unc = self.modality_corrector_forward(
            self.semantic_blocks(
                inputs_embeds=bert_hiddens,
                position_ids=zero_position_ids,
                attention_mask=attention_mask
            )[0],
            input_ids
        )
        uncertainty_metrics.extend(semantic_unc)

        graphic_logits, graphic_unc = self.modality_corrector_forward(
            self.graphic_blocks(
                inputs_embeds=res_hiddens + bert_hiddens,
                position_ids=zero_position_ids,
                attention_mask=attention_mask
            )[0],
            input_ids
        )
        uncertainty_metrics.extend(graphic_unc)

        phonetic_logits, pho_unc = self.modality_corrector_forward(
            self.phonetic_blocks(
                inputs_embeds=pho_hiddens + bert_hiddens,
                position_ids=zero_position_ids,
                attention_mask=attention_mask
            )[0],
            input_ids
        )
        uncertainty_metrics.extend(pho_unc)

        modal_weights = {}
        if self.modal_weighting_ablation:
            hiddens = bert_hiddens + res_hiddens + pho_hiddens
        else:
            modal_weight = self.modal_scale(self.modal_weight_act(
                self.modal_weight_network(*uncertainty_metrics)
            ))
            modal_weights['semantic_modal_weight'] = modal_weight[:, :, 0:1]
            modal_weights['graphic_modal_weight'] = modal_weight[:, :, 1:2]
            modal_weights['phonetic_modal_weight'] = modal_weight[:, :, 2:3]

            hiddens = bert_hiddens * modal_weights['semantic_modal_weight'] \
                      + res_hiddens * modal_weights['graphic_modal_weight'] \
                      + pho_hiddens * modal_weights['phonetic_modal_weight']

        joint_logits, joint_unc = self.modality_corrector_forward(
            self.joint_blocks(
                inputs_embeds=hiddens,
                position_ids=zero_position_ids,
                attention_mask=attention_mask
            )[0],
            input_ids
        )
        uncertainty_metrics.extend(joint_unc)

        logits_weights = {}
        if self.ensemble_weighting_ablation:
            logits = semantic_logits + graphic_logits + phonetic_logits + joint_logits
        else:
            weight = torch.softmax(
                self.correct_weight_network(*uncertainty_metrics),
                dim=-1
            )
            logits_weights['semantic_logits_weight'] = weight[:, :, 0:1]
            logits_weights['graphic_logits_weight'] = weight[:, :, 1:2]
            logits_weights['phonetic_logits_weight'] = weight[:, :, 2:3]
            logits_weights['joint_logits_weight'] = weight[:, :, 3:4]
            logits = semantic_logits * logits_weights['semantic_logits_weight'] \
                     + graphic_logits * logits_weights['graphic_logits_weight'] \
                     + phonetic_logits * logits_weights['phonetic_logits_weight'] \
                     + joint_logits * logits_weights['joint_logits_weight']

        return {
            'logits': logits,
            'semantic_logits': semantic_logits,
            'graphic_logits': graphic_logits,
            'phonetic_logits': phonetic_logits,
            'joint_logits': joint_logits,

            # weights outputs:
            # **modal_weights,
            # **logits_weights,
            # uncertainty metrics outputs:
            # 'uncertainty_metrics': uncertainty_metrics
        }

    def modality_corrector_forward(self, hiddens, input_ids):
        logits = self.classifier(self.dropout(hiddens))
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
        return logits, [correct_entropy, detect_entropy]


class UGReaLiSeUMFAblation(UGReaLiSe):
    def __init__(self, config):
        super().__init__(config, modal_weighting_ablation=True)


class UGReaLiSeUCAblation(UGReaLiSe):
    def __init__(self, config):
        super().__init__(config, ensemble_weighting_ablation=True)

