""" Base model with common functionality  """

import torch
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, model_config):
        super(BaseModel, self).__init__()
        self.num_steps_integral_loss = model_config.get(
            'num_steps_integral_loss', 20)
        self.add_bos = model_config.get('add_bos', False)
        self.hidden_size = model_config.get('hidden_size', 64)
        self.num_event_types_no_pad = model_config['num_event_types_no_pad']  # not include [PAD], [BOS], [EOS]
        self.num_event_types_pad = model_config['num_event_types_pad']  # include [PAD], [BOS], [EOS]
        self.event_pad_index = model_config['event_pad_index']
        self.eps = torch.finfo(torch.float32).eps

        self.layer_event_emb = nn.Embedding(self.num_event_types_pad,  # have padding
                                            self.hidden_size,
                                            padding_idx=self.event_pad_index)

    @staticmethod
    def generate_model_from_config(model_config):
        model_name = model_config.get('name')

        cls_list = []
        cls_list.extend(BaseModel.__subclasses__())
        idx = 0
        while idx < len(cls_list):
            subcls = cls_list[idx]
            if subcls.__name__ == model_name:
                return subcls(model_config)
            cls_list.extend(subcls.__subclasses__())
            idx += 1

        raise RuntimeError('No model named ' + model_name)

    def compute_loglik(self, batch):
        raise NotImplementedError