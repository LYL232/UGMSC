import torch
from torch import nn
from transformers import BertModel, BertConfig
from copy import deepcopy
import math


class GRU(nn.Module):
    def __init__(self, input_num, hidden_num):
        super(GRU, self).__init__()
        self.hidden_size = hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        # self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden = self.GRU_layer(x)
        # x = self.output_linear(x)
        return x, self.hidden


def weights_init(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            print(name)
            torch.nn.init.normal_(param, mean=0, std=0.02)
        else:
            nn.init.zeros_(param)


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


class UG_Bert_PLOME(nn.Module):
    def __init__(
            self,
            model_path,
            num_class,
            pyid2seq,
            skid2seq,
            py_dim,
            max_sen_len,
            zi_py_matrix,
            corrector_layers: int,
            dropout
    ):
        super().__init__()
        config = BertConfig.from_pretrained(model_path)
        self.model = BertModel(config)
        self.num_class = num_class  # 21128
        self.embed_size = self.model.config.hidden_size
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size
        self.dropout = nn.Dropout(dropout)
        self.pyid2seq = pyid2seq
        self.skid2seq = skid2seq
        self.PYLEN = 4
        self.SKLEN = 10
        self.PYDIM = py_dim
        self.MAX_SEN_LEN = max_sen_len
        self.zi_py_matrix = zi_py_matrix
        self.pinyin_num = 430

        self.py_ebd = torch.nn.Embedding(30, self.PYDIM)
        self.sk_ebd = torch.nn.Embedding(7, self.PYDIM)
        # self.py_ebd.apply(weights_init)
        # self.sk_ebd.apply(weights_init)
        self.gru_py = GRU(input_num=self.PYDIM, hidden_num=self.hidden_size)
        self.gru_sk = GRU(input_num=self.PYDIM, hidden_num=self.hidden_size)
        # self.gru_py.apply(weights_init)
        # self.gru_sk.apply(weights_init)
        self.hanzi_linear = nn.Linear(self.hidden_size, self.num_class)
        self.pinyin_linear = nn.Linear(self.hidden_size, self.pinyin_num)
        self.softmax = nn.Softmax(dim=-1)

        # addition
        block_config = deepcopy(config)
        block_config.num_hidden_layers = corrector_layers
        self.semantic_blocks = BertModel(block_config)
        self.phonetic_blocks = BertModel(block_config)
        self.graphic_blocks = BertModel(block_config)

        self.modal_weight_network = NumericalMLP(
            in_features=6,  # (改正有检测和改正两个熵，共3个corrector：2 * 3)
            hidden_size=config.hidden_size,
            out_features=3,
            dropout=config.hidden_dropout_prob
        )
        self.modal_scale = PositiveScale(3, 2.0)

        self.modal_weight_act = nn.Sigmoid()

        self.correct_weight_network = NumericalMLP(
            in_features=10,  # (改正有检测和改正两个熵，共4个corrector：2 * 4, + 2个拼音的检测和改正熵)
            hidden_size=config.hidden_size,
            out_features=4,
            dropout=config.hidden_dropout_prob
        )

        self.max_hanzi_ent = math.log(self.vocab_size)
        self.max_pinyin_ent = math.log(self.pinyin_num)
        self.max_detect_ent = math.log(2)
        self.eps = 1e-12

        self.semantic_blocks.embeddings.word_embeddings.weight = self.model.embeddings.word_embeddings.weight
        self.graphic_blocks.embeddings.word_embeddings.weight = self.model.embeddings.word_embeddings.weight
        self.phonetic_blocks.embeddings.word_embeddings.weight = self.model.embeddings.word_embeddings.weight

    def get_entropy(self, logits, ids, max_ent):
        prob = torch.softmax(logits.detach(), dim=-1)
        correct_entropy = -1 * torch.sum(prob * torch.log(prob), dim=-1, keepdim=True) / max_ent
        input_ids_prob = torch.gather(prob, dim=-1, index=ids)
        err_prob = 1 - input_ids_prob
        detect_entropy = -1 * (
                input_ids_prob * torch.log(input_ids_prob + self.eps) +
                err_prob * torch.log(err_prob + self.eps)
        ) / self.max_detect_ent
        return [correct_entropy, detect_entropy]

    def modality_forward(self, hiddens, input_ids):
        predict_logits = self.dropout(hiddens)
        logits_zi = self.hanzi_linear(predict_logits)
        zi_ent = self.get_entropy(logits_zi, input_ids, self.max_hanzi_ent)
        return logits_zi, [*zi_ent]

    def joint_forward(self, hiddens, input_ids, pinyin_ids):
        predict_logits = self.dropout(hiddens)
        logits_zi = self.hanzi_linear(predict_logits)
        logits_py = self.pinyin_linear(predict_logits)
        zi_ent = self.get_entropy(logits_zi, input_ids, self.max_hanzi_ent)
        py_ent = self.get_entropy(logits_py, pinyin_ids, self.max_pinyin_ent)
        return logits_zi, logits_py, [*zi_ent, *py_ent]

    def lookup_py(self, ID2SEQ, sen_pyids):
        py_seqs = [ID2SEQ[py_id] for py_id in sen_pyids]
        return py_seqs

    def joint_embeddings(self, input_ids, position_ids, weights, *embs):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.model.embeddings.position_embeddings(position_ids)
        embeddings = position_embeddings
        for i, emb in enumerate(embs):
            embeddings += weights[:, :, i:i + 1] * emb
        embeddings = self.model.embeddings.LayerNorm(embeddings)
        embeddings = self.model.embeddings.dropout(embeddings)
        return embeddings

    @staticmethod
    def block_emb(blocks, input_ids, block_emb, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = blocks.embeddings.position_embeddings(position_ids)
        embeddings = block_emb + position_embeddings
        embeddings = blocks.embeddings.LayerNorm(embeddings)
        embeddings = blocks.embeddings.dropout(embeddings)
        return embeddings

    def forward(self, input_ids, pinyin_ids, stroke_ids, device, position_ids=None, attention_mask=None,
                is_training=False):
        # input_ids : [batch size, sequence length]
        # attention_mask : [batch size, sequence length]
        input_shape = input_ids.size()
        b, s = input_shape[0], input_shape[1]

        # 对应 BertModel 中的 get_extended_attention_mask(attention_mask, input_shape, device)
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(torch.float32)
        attention_mask = (1.0 - attention_mask) * -10000.0

        # pinyin_stroke Embedding
        pinyin_ids = torch.reshape(pinyin_ids, shape=[-1])
        stroke_ids = torch.reshape(stroke_ids, shape=[-1])
        py_seqs = torch.LongTensor(self.lookup_py(self.pyid2seq, pinyin_ids)).to(device)
        sk_seqs = torch.LongTensor(self.lookup_py(self.skid2seq, stroke_ids)).to(device)
        py_seq_emb = self.py_ebd(py_seqs)
        py_seq_emb = torch.reshape(py_seq_emb, shape=[-1, self.PYLEN, self.PYDIM])
        sk_seq_emb = self.sk_ebd(sk_seqs)
        sk_seq_emb = torch.reshape(sk_seq_emb, shape=[-1, self.SKLEN, self.PYDIM])
        _, output_py = self.gru_py(py_seq_emb)
        _, output_sk = self.gru_sk(sk_seq_emb)

        output_py = torch.reshape(output_py, [-1, self.MAX_SEN_LEN, self.hidden_size])
        output_sk = torch.reshape(output_sk, [-1, self.MAX_SEN_LEN, self.hidden_size])
        words_embeddings = self.model.embeddings.word_embeddings(input_ids)
        expand_input_ids = input_ids.view((b, s, 1))
        expand_pinyin_ids = pinyin_ids.view((b, s, 1))

        uncertainty_metrics = []

        emb = self.block_emb(
            blocks=self.semantic_blocks,
            input_ids=input_ids,
            block_emb=words_embeddings
        )
        semantic_hiddens = self.semantic_blocks.encoder(emb, attention_mask)[0]
        semantic_zi_logits, semantic_unc = \
            self.modality_forward(semantic_hiddens, expand_input_ids)
        uncertainty_metrics.extend(semantic_unc)

        emb = self.block_emb(
            blocks=self.phonetic_blocks,
            input_ids=input_ids,
            block_emb=words_embeddings + output_py
        )
        phonetic_hiddens = self.phonetic_blocks.encoder(emb, attention_mask)[0]
        phonetic_zi_logits, phonetic_unc = self.modality_forward(phonetic_hiddens, expand_input_ids)
        uncertainty_metrics.extend(phonetic_unc)

        emb = self.block_emb(
            blocks=self.graphic_blocks,
            input_ids=input_ids,
            block_emb=words_embeddings + output_sk
        )
        graphic_hiddens = self.graphic_blocks.encoder(emb, attention_mask)[0]
        graphic_zi_logits, graphic_unc = self.modality_forward(graphic_hiddens, expand_input_ids)
        uncertainty_metrics.extend(graphic_unc)

        modal_weight = self.modal_scale(self.modal_weight_act(
            self.modal_weight_network(*uncertainty_metrics)
        ))

        embedding_output = self.joint_embeddings(
            input_ids,
            position_ids,
            modal_weight,
            words_embeddings,
            output_py,
            output_sk
        )

        joint_hiddens = self.model.encoder(embedding_output, attention_mask)[0]
        joint_zi_logits, logits_pinyin, joint_unc = self.joint_forward(
            joint_hiddens, expand_input_ids, expand_pinyin_ids
        )
        uncertainty_metrics.extend(joint_unc)

        weight = torch.softmax(self.correct_weight_network(*uncertainty_metrics), dim=-1)

        logits_hanzi = semantic_zi_logits * weight[:, :, 0:1] \
                       + graphic_zi_logits * weight[:, :, 1:2] \
                       + phonetic_zi_logits * weight[:, :, 2:3] \
                       + joint_zi_logits * weight[:, :, 3:4]

        # fusion output
        prob_fusion = torch.zeros(1, 1).to(device)
        if not is_training:
            prob_fusion = torch.matmul(self.softmax(logits_pinyin),
                                       torch.t(torch.FloatTensor(self.zi_py_matrix).to(device)))
            prob_fusion = prob_fusion * self.softmax(logits_hanzi)

        return {
            'logits_hanzi': logits_hanzi,
            'logits_pinyin': logits_pinyin,
            'semantic_zi_logits': semantic_zi_logits,
            'graphic_zi_logits': graphic_zi_logits,
            'phonetic_zi_logits': phonetic_zi_logits,
            'joint_zi_logits': joint_zi_logits,
            'prob_fusion': prob_fusion
        }
