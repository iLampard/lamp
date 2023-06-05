import torch

from ..tpp.att_nhp import AttNHP


class AttNHPEBMTPP(AttNHP):
    def __init__(self,
                 model_config):
        super(AttNHPEBMTPP, self).__init__(model_config)
        self.out_ffn_layer = torch.nn.Sequential(
            torch.nn.Linear(self.d_model * self.n_head, self.d_model),
            torch.nn.GELU(),
            torch.nn.Linear(self.d_model, 1)
        )

    def forward(
            self,
            time_seqs,
            time_delta_seqs,
            type_seqs,
            batch_non_pad_mask,
            attention_mask,
            type_mask
    ):
        # shape -> [..., n_seqs, seq_len]

        rnn_seq_output = []
        num_seqs = time_seqs.size(-2)
        for i in range(num_seqs):
            batch_i = (type_seqs[:, i, :],
                       time_seqs[:, i, :],
                       batch_non_pad_mask[:, i, :],
                       attention_mask[:, i, :, :])
            # [batch_size, seq_len, hidden_size]
            rnn_seq_output_i = super().forward_along_seqs(*batch_i)

            # [batch_size, hidden_size]
            rnn_seq_output_i = self.get_logits_at_last_step(rnn_seq_output_i, batch_non_pad_mask[:, i, :])

            rnn_seq_output.append(rnn_seq_output_i)

        # shape -> [batch_size, n_seqs, hidden_dim]
        rnn_logit = torch.stack(rnn_seq_output, dim=1)

        # shape -> [..., n_seqs, 1]
        logit = self.out_ffn_layer(rnn_logit)

        # shape -> [..., n_seqs]
        logit = logit[..., 0]

        return logit
