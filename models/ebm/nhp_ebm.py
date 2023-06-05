import torch
from torch import nn

from models.tpp.nhp import NHP


class NHPDisc(NHP):
    def __init__(self, model_config):
        super(NHPDisc, self).__init__(model_config)

        # prediction for discriminator
        self.discriminator_prediction_layer = torch.nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.discriminator_loss = nn.CrossEntropyLoss(reduction='mean')

    def predict_as_discriminator(self, logits):
        logits = self.discriminator_prediction_layer(logits)
        # use softmax to scale values to probability
        # logits = torch.softmax(logits, dim=-1)

        return logits

    @staticmethod
    def get_logits_at_last_step(logits, batch_non_pad_mask, sample_len=None):
        # ref: https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        seq_len = batch_non_pad_mask.sum(dim=1)
        select_index = seq_len - 1 if sample_len is None else seq_len - 1 - sample_len
        # [batch_size, hidden_dim]
        select_index = select_index.unsqueeze(1).repeat(1, logits.size(-1))
        # [batch_size, 1, hidden_dim]
        select_index = select_index.unsqueeze(1)
        # [batch_size, hidden_dim]
        last_logits = torch.gather(logits, dim=1, index=select_index).squeeze(1)
        return last_logits
