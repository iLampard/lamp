import math

import numpy as np
import torch
from torch import nn

from .base_model import BaseModel
from .layers import EncoderLayer, MultiHeadAttention
from .thinning import  EventSampler

# ref: https://github.com/yangalan123/anhp-andtt/blob/master/anhp/model/xfmr_nhp_fast.py

class AttNHP(BaseModel):
    def __init__(self, model_config):
        # d_model, n_layers, n_head, dropout, d_time, use_norm=False,
        # sharing_param_layer=False):
        # d_inner only used if we want to add feedforward
        super(AttNHP, self).__init__(model_config)
        self.d_model = model_config['hidden_size']
        self.d_time = model_config['time_emb_size']
        self.use_norm = model_config['use_ln']

        self.div_term = torch.exp(torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)).reshape(1, 1,
                                                                                                                -1)
        self.n_layers = model_config['num_layers']
        self.n_head = model_config['num_heads']
        self.sharing_param_layer = model_config['sharing_param_layer']
        self.dropout = model_config['dropout']
        self.mc_num_sample_per_step = model_config['mc_num_sample_per_step']

        if not self.sharing_param_layer:
            self.heads = []
            for i in range(self.n_head):
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
                            for _ in range(self.n_layers)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)
        else:
            self.heads = []
            for i in range(self.n_head):
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
                            for _ in range(0)
                        ]
                    )
                )
            self.heads = nn.ModuleList(self.heads)

        if self.use_norm:
            self.norm = nn.LayerNorm(self.d_model)
        self.inten_linear = nn.Linear(self.d_model * self.n_head, self.num_event_types_no_pad)
        self.softplus = nn.Softplus()
        self.layer_intensity = nn.Sequential(self.inten_linear, self.softplus)
        self.eps = torch.finfo(torch.float32).eps
        # self.add_bos = dataset.add_bos
        self.add_bos = True

        self.event_sampler = EventSampler(model_config['thinning_params']['num_samples'],
                                          model_config['thinning_params']['num_exp'])

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
        for head_i in range(self.n_head):
            cur_layer_ = init_cur_layer_
            for layer_i in range(self.n_layers):
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

    def forward_along_seqs(self, event_seqs, time_seqs, batch_non_pad_mask, attention_mask, extra_times=None):
        tem_enc = self.compute_temporal_embedding(time_seqs)
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)
        enc_input = torch.tanh(self.layer_event_emb(event_seqs))
        init_cur_layer_ = torch.zeros_like(enc_input)
        layer_mask = (torch.eye(attention_mask.size(1)) < 1).unsqueeze(0).expand_as(attention_mask).to(
            attention_mask.device)

        if extra_times is None:
            tem_enc_layer = tem_enc
        else:
            tem_enc_layer = self.compute_temporal_embedding(extra_times)
            tem_enc_layer *= batch_non_pad_mask.unsqueeze(-1)

        # batch_size * (seq_len) * (2 * seq_len)
        _combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # batch_size * (2 * seq_len) * (2 * seq_len)
        contextual_mask = torch.cat([attention_mask, torch.ones_like(layer_mask)], dim=-1)
        _combined_mask = torch.cat([contextual_mask, _combined_mask], dim=1)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_enc_layer, enc_input, _combined_mask,
                                       batch_non_pad_mask)
        return cur_layer_

    def forward(self, batch, return_loss: bool = False):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask, _, _ = batch
        # 1. compute event-loglik
        enc_out = self.forward_along_seqs(event_seq[:, :-1], time_seq[:, :-1], batch_non_pad_mask[:, 1:],
                                          attention_mask[:, 1:, :-1], time_seq[:, 1:])

        loss = 0
        if return_loss:
            enc_inten = self.layer_intensity(enc_out)
            # original: 1->1, 2->2
            # event_lambdas = torch.sum(enc_inten * type_mask, dim=2) + self.eps
            # now: 1->2, 2->3
            event_lambdas = torch.sum(enc_inten * type_mask[:, 1:], dim=2) + self.eps
            # in case event_lambdas == 0
            # event_lambdas.masked_fill_(~batch_non_pad_mask, 1.0)
            event_lambdas.masked_fill_(~batch_non_pad_mask[:, 1:], 1.0)

            event_ll = torch.log(event_lambdas)

            # 2. compute non-event-loglik (using MC sampling to compute integral)
            # num_samples = 200
            num_samples = self.mc_num_sample_per_step
            # 2.1 sample times
            # 2.2 compute intensities at sampled times
            # due to GPU memory limitation, we may not be able to compute all intensities at all sampled times,
            # step gives the batch size w.r.t how many sampled times we should process at each batch
            step = self.num_steps_integral_loss
            if not self.add_bos:
                extended_time_seq = torch.cat([torch.zeros(time_seq.size(0), 1).to(time_seq.device), time_seq], dim=-1)
                diff_time = (time_seq[:, :] - extended_time_seq[:, :-1]) * batch_non_pad_mask[:, :]
                temp_time = diff_time.unsqueeze(0) * \
                            torch.rand([num_samples, *diff_time.size()], device=time_seq.device)
                temp_time += extended_time_seq[:, :-1].unsqueeze(0)
                all_lambda = self._compute_intensities_fast(event_seq, time_seq, batch_non_pad_mask, attention_mask,
                                                            temp_time, step)
            else:
                # why non_pad_mask start from 1?
                # think about a simple case: [e] [e] [pad] (non_pad_mask: 1 1 0)
                # you want to compute the first interval only, so if you use non_pad_mask[:, :-1] (1, 1),
                # you will compute both the first and the second intervals!
                diff_time = (time_seq[:, 1:] - time_seq[:, :-1]) * batch_non_pad_mask[:, 1:]
                temp_time = diff_time.unsqueeze(0) * \
                            torch.rand([num_samples, *diff_time.size()], device=time_seq.device)
                temp_time += time_seq[:, :-1].unsqueeze(0)
                # for interval computation, we will never use the last event -- that is why we have -1 in
                # event_seq, time_seq, attention_mask
                all_lambda = self._compute_intensities_fast(event_seq[:, :-1], time_seq[:, :-1],
                                                            batch_non_pad_mask[:, 1:],
                                                            attention_mask[:, 1:, :-1],
                                                            temp_time, step)

            # sum over type_events, then sum over sampled times
            all_lambda = all_lambda.sum(dim=-1)

            # 2.3 compute the empirical expectation of the summation
            all_lambda = all_lambda.sum(dim=0) / num_samples
            non_event_ll = all_lambda * diff_time

            loss = -torch.sum(event_ll - non_event_ll)
            num_events = (event_seq[:, 1:] < self.event_pad_index).sum()
            loss /= num_events

        if return_loss:
            ret_tuple = (loss, enc_out)
        else:
            ret_tuple = (enc_out)
        return ret_tuple

    def _compute_intensities_fast(self,
                                  event_seq,
                                  time_seq,
                                  batch_non_pad_mask,
                                  attention_mask,
                                  temp_time,
                                  step=20):
        # fast version, can only use in log-likelihood computation
        # assume we will sample the same number of times in each interval of the event_seqs
        all_lambda = []
        batch_size = event_seq.size(0)
        seq_len = event_seq.size(1)
        num_samples = temp_time.size(0)
        for i in range(0, num_samples, step):
            _extra_time = temp_time[i: i + step, :, :]
            _step = _extra_time.size(0)
            _extra_time = _extra_time.reshape(_step * batch_size, -1)
            _types = event_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _times = time_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _batch_non_pad_mask = batch_non_pad_mask.unsqueeze(0).expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _attn_mask = attention_mask.unsqueeze(0).expand(_step, -1, -1, -1).reshape(_step * batch_size, seq_len,
                                                                                       seq_len)
            _enc_output = self.forward_along_seqs(_types,
                                                  _times,
                                                  _batch_non_pad_mask,
                                                  _attn_mask,
                                                  _extra_time)
            all_lambda.append(self.layer_intensity(_enc_output).reshape(_step, batch_size, seq_len, -1))
        all_lambda = torch.cat(all_lambda, dim=0)
        return all_lambda

    def compute_intensities_at_sampled_times(self, event_seq, time_seq, sampled_times):
        # Assumption: all the sampled times are distributed [time_seq[...,-1], next_event_time]
        # used for thinning algorithm
        num_batches = event_seq.size(0)
        seq_len = event_seq.size(1)
        assert num_batches == 1, "Currently, no support for batch mode (what is a good way to do batching in thinning?)"
        if num_batches == 1 and num_batches < sampled_times.size(0):
            _sample_size = sampled_times.size(0)
            # multiple sampled_times
            event_seq = event_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
            time_seq = time_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
            num_batches = event_seq.size(0)
        assert (time_seq[:, -1:] <= sampled_times).all(), "sampled times must occur not earlier than last events!"
        num_samples = sampled_times.size(1)

        # 1. prepare input embeddings for "history"
        tem_enc = self.compute_temporal_embedding(time_seq)
        # enc_input = torch.tanh(self.Emb(event_seq))
        enc_input = torch.tanh(self.layer_event_emb(event_seq))
        init_cur_layer_ = torch.zeros((sampled_times.size(0), sampled_times.size(1), enc_input.size(-1))).to(
            sampled_times.device)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        tem_layer_ = self.compute_temporal_embedding(sampled_times)

        # 2. prepare attention mask
        attention_mask = torch.ones((num_batches, seq_len + num_samples, seq_len + num_samples)).to(event_seq.device)
        # attention_mask[:, :seq_len, :seq_len] = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0).cuda()
        attention_mask[:, :seq_len, :seq_len] = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).unsqueeze(0).to(
            event_seq.device)
        # by default, regard all_sampled times to be equal to the last_event_time
        # recall that we use 1 for "not attending", 0 for "attending"
        # t_i == sampled_t
        attention_mask[:, seq_len:, :seq_len - 1] = 0
        # t_i < sampled_t
        attention_mask[:, seq_len:, seq_len - 1][time_seq[:, -1:] < sampled_times] = 0
        attention_mask[:, seq_len:, seq_len:] = (torch.eye(num_samples) < 1).unsqueeze(0).to(event_seq.device)
        cur_layer_ = self.forward_pass(init_cur_layer_, tem_enc, tem_layer_, enc_input, attention_mask)

        sampled_intensities = self.softplus(self.inten_linear(cur_layer_))

        return sampled_intensities

    @staticmethod
    def get_logits_at_last_step(logits, batch_non_pad_mask, sample_len=None):
        # ref: https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        seq_len = batch_non_pad_mask.sum(dim=1)
        select_index = seq_len - 1 if sample_len is None else seq_len - 1 - sample_len
        # [..., hidden_dim]
        select_index = select_index.unsqueeze(1).repeat(1, logits.size(-1))
        # [..., 1, hidden_dim]
        select_index = select_index.unsqueeze(1)
        # [..., hidden_dim]
        last_logits = torch.gather(logits, dim=1, index=select_index).squeeze(1)
        return last_logits

    def inference(self, batch):

        # thinning can only run in single instance mode, not in batch mode
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask, \
            seq_idx, original_idx = batch

        if time_seq.size(1) == 0 or time_seq.size(1) == 1 or original_idx.size(1) == 0:
            return []

        pred_res = []

        num_batch = time_seq.size(0)
        assert num_batch == 1

        for i in range(num_batch):
            rst = []
            _time_seq, _event_seq = time_seq[i][batch_non_pad_mask[i]], event_seq[i][batch_non_pad_mask[i]]
            seq_len = _time_seq.size(0)
            duration = _time_seq[-1].item() + np.finfo(float).eps
            num_sub = seq_len - 1
            range_start = max(original_idx[i][0], 1) # some test seq starts from 0-th position, we should avoid this
            range_end = original_idx[i][-1]
            # only evaluate target index event
            for j in range(range_start-1, range_end - 1):
                next_event_name, next_event_time = _event_seq[j + 1].item(), _time_seq[j + 1].item()
                current_event_name, current_event_time = _event_seq[j].item(), _time_seq[j].item()
                time_last_event = _time_seq[j].item()
                next_event_dtime = next_event_time - time_last_event
                avg_future_dtime = (duration - time_last_event) / (num_sub - j)
                look_ahead = max(next_event_dtime, avg_future_dtime)
                boundary = time_last_event + 4 * look_ahead
                _event_prefix, _time_prefix = _event_seq[:j + 1].unsqueeze(0), _time_seq[:j + 1].unsqueeze(
                    0)
                accepted_times, weights = self.event_sampler.draw_next_time(
                    [[_event_prefix, _time_prefix],
                     time_last_event, boundary, self.compute_intensities_at_sampled_times]
                )
                time_uncond = float(torch.sum(accepted_times * weights))
                dtime_uncond = time_uncond - time_last_event
                intensities_at_times = self.compute_intensities_at_sampled_times(
                    _event_prefix, _time_prefix,
                    _time_seq[j + 1].reshape(1, 1)
                )[0, 0]
                top_ids = torch.argsort(intensities_at_times, dim=0, descending=True)
                # since we use int to represent event names already
                top_event_names = [int(top_i) for top_i in top_ids]
                rst.append(
                    (
                        time_uncond, dtime_uncond, top_event_names,
                        next_event_time, next_event_dtime, next_event_name, intensities_at_times
                    )
                )

                pred_dict_ = dict()
                pred_dict_['pred_dtime']  = dtime_uncond
                pred_dict_['label_dtime'] = next_event_dtime
                pred_dict_['pred_type_score'] = intensities_at_times.detach().cpu().numpy()
                pred_dict_['label_type'] = next_event_name
                pred_dict_['seq_idx'] = seq_idx.detach().cpu().numpy()[0][0] # batch_size=1, seq_idx is the same for all the event in the same seq
                pred_dict_['original_idx'] = j+1

                pred_res.append(pred_dict_)
        return pred_res
