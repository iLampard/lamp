import math

import torch
from torch import nn

from ..tpp.layers import EncoderLayer, MultiHeadAttention


class AttNHPEBM(torch.nn.Module):
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 num_noise_samples: int,
                 embedding_dim: int,
                 d_model: int,
                 d_time: int = 32,
                 num_heads: int = 2,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 sharing_param_layer: bool = False,
                 use_ln: bool = False):
        super(AttNHPEBM, self).__init__()

        self.num_noise_samples = num_noise_samples
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_time = d_time
        self.dropout = dropout
        self.num_layers = num_layers
        self.sharing_param_layer = sharing_param_layer
        self.entity_embedding_layer = torch.nn.Embedding(num_entities, embedding_dim=embedding_dim)
        self.relation_embedding_layer = torch.nn.Embedding(num_relations, embedding_dim=embedding_dim)

        self.div_term = torch.exp(torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)).reshape(1, 1,
                                                                                                                -1)
        self.use_norm = use_ln
        self.input_layer = nn.Linear(embedding_dim * 3, self.d_model)

        self.heads = []
        for i in range(self.num_heads):
            self.heads.append(
                nn.ModuleList(
                    [EncoderLayer(
                        self.d_model + self.d_time,
                        MultiHeadAttention(1, self.d_model + self.d_time, self.d_model, self.dropout,
                                           output_linear=False),
                        # PositionwiseFeedForward(d_model + d_time, d_inner, dropout),
                        use_residual=False,
                        dropout=self.dropout
                    )
                        for _ in range(self.num_layers)
                    ]
                )
            )
        self.heads = nn.ModuleList(self.heads)

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)

        self.out_ffn_layer = torch.nn.Sequential(
            torch.nn.Linear(d_model*num_heads, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, 1)
        )

        self.discriminator_loss = nn.CrossEntropyLoss(reduction='mean')

    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe

    def forward_pass(self, init_cur_layer_, tem_enc, tem_enc_layer, enc_input, combined_mask, batch_non_pad_mask=None):
        cur_layers = []
        seq_len = enc_input.size(1)
        for head_i in range(self.num_heads):
            cur_layer_ = init_cur_layer_
            for layer_i in range(self.num_layers):
                layer_ = torch.cat([cur_layer_, tem_enc_layer], dim=-1)
                _combined_input = torch.cat([enc_input, layer_], dim=1)
                if self.sharing_param_layer:
                    enc_layer = self.heads[head_i][0]
                else:
                    enc_layer = self.heads[head_i][layer_i]
                enc_output = enc_layer(
                    _combined_input,
                    combined_mask
                )
                if batch_non_pad_mask is not None:
                    _cur_layer_ = enc_output[:, seq_len:, :] * (batch_non_pad_mask.unsqueeze(-1))
                else:
                    _cur_layer_ = enc_output[:, seq_len:, :]

                # add residual connection
                cur_layer_ = torch.tanh(_cur_layer_) + cur_layer_
                enc_input = torch.cat([enc_output[:, :seq_len, :], tem_enc], dim=-1)
                # non-residual connection
                # cur_layer_ = torch.tanh(_cur_layer_)

                # enc_output *= _combined_non_pad_mask.unsqueeze(-1)
                # layer_ = torch.tanh(enc_output[:, enc_input.size(1):, :])
                if self.use_norm:
                    cur_layer_ = self.norm(cur_layer_)
            cur_layers.append(cur_layer_)
        cur_layer_ = torch.cat(cur_layers, dim=-1)

        return cur_layer_

    def att_nhp_layer(self, time_seqs, rnn_input, batch_non_pad_mask, attention_mask):
        tem_enc = self.compute_temporal_embedding(time_seqs)
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)
        enc_input = rnn_input
        init_cur_layer_ = torch.zeros_like(enc_input)
        layer_mask = (torch.eye(attention_mask.size(1)) < 1).unsqueeze(0).expand_as(attention_mask).to(
            attention_mask.device)

        tem_enc_layer = tem_enc
        # batch_size * (seq_len) * (2 * seq_len)
        _combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # batch_size * (2 * seq_len) * (2 * seq_len)
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask)], dim=-1)
        _combined_mask = torch.cat([contextual_mask, _combined_mask], dim=1)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_enc_layer, enc_input, _combined_mask,
                                       batch_non_pad_mask)

        return cur_layer_

    def make_att_mask(self, seq_time):
        batch_size, num_seq, seq_len = seq_time.size(0), seq_time.size(1), seq_time.size(2)
        batch_seq_pad_mask_ = seq_time.eq(0)
        # shape -> [batch_size, num_seq, seq_len, seq_len]
        attention_key_pad_mask = batch_seq_pad_mask_.unsqueeze(2).expand(batch_size, num_seq, seq_len, -1)
        # shape -> [batch_size, num_seq, seq_len, seq_len]
        subsequent_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=0
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, num_seq, -1, -1)
        attention_mask = subsequent_mask | attention_key_pad_mask.bool()

        return attention_mask

    def forward(
            self,
            noise_seq_subject,
            noise_seq_object,
            noise_seq_relation,
            noise_seq_time,
    ):
        # shape -> [..., n_seqs, seq_len, embedding_dim]
        noise_seq_subject_emb = self.entity_embedding_layer(noise_seq_subject)
        noise_seq_object_emb = self.entity_embedding_layer(noise_seq_object)
        noise_seq_relation_emb = self.relation_embedding_layer(noise_seq_relation)

        # shape -> [..., n_seqs, seq_len]
        batch_non_pad_mask = noise_seq_time > 0

        # shape -> [..., n_seqs, seq_len]
        attention_mask = self.make_att_mask(noise_seq_time)

        rnn_input = torch.cat([
            noise_seq_subject_emb,
            noise_seq_object_emb,
            noise_seq_relation_emb,
        ], dim=-1)

        rnn_input = self.input_layer(rnn_input)

        rnn_seq_output = []
        num_seqs = noise_seq_time.size(-2)
        for i in range(num_seqs):
            rnn_seq_output_i = self.att_nhp_layer(noise_seq_time[:, i, :],  # [batch_size, seq_len]
                                                  rnn_input[:, i, :, :],  # [batch_size, seq_len, hidden_size]
                                                  batch_non_pad_mask[:, i, :],  # [batch_size, seq_len]
                                                  attention_mask[:, i, :, :])  # [batch_size, seq_len, seq_len]

            rnn_seq_output.append(rnn_seq_output_i)

        # shape -> [..., n_seqs, seq_len, hidden_dim]
        rnn_seq_output = torch.stack(rnn_seq_output, dim=1)

        # shape -> [..., n_seqs, 1]
        logit = self.out_ffn_layer(rnn_seq_output[..., :, -1, :])

        # shape -> [..., n_seqs]
        logit = logit[..., 0]

        return logit
