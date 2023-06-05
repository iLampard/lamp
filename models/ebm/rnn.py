
import torch


class RNNEbm(torch.nn.Module):
    def __init__(
            self,
            num_entities: int,
            num_relations: int,
            num_noise_samples: int,
            embedding_dim: int,
            num_cells: int,
            num_layers: int = 2,
            dropout_rate: float = 0.1
    ):
        super().__init__()
        self.num_noise_samples = num_noise_samples

        self.entity_embedding_layer = torch.nn.Embedding(num_entities, embedding_dim=embedding_dim)
        self.relation_embedding_layer = torch.nn.Embedding(num_relations, embedding_dim=embedding_dim)

        self.in_ffn_layer = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 3, num_cells),
            torch.nn.GELU(),
            torch.nn.Linear(num_cells, num_cells)
        )
        self.out_ffn_layer = torch.nn.Sequential(
            torch.nn.Linear(num_cells, num_cells),
            torch.nn.GELU(),
            torch.nn.Linear(num_cells, 1)
        )

        self.rnn_layer = RNN(
            feat_dim=num_cells,
            num_cells=num_cells,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            cell_type='lstm',
        )

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

        mask = noise_seq_time > 0

        rnn_input = torch.cat([
            noise_seq_subject_emb,
            noise_seq_object_emb,
            noise_seq_relation_emb,
        ], dim=-1)

        rnn_input = rnn_input * mask[..., None]

        rnn_input = self.in_ffn_layer(rnn_input)

        rnn_seq_output, _ = self.rnn_layer(rnn_input)

        # shape -> [..., n_seqs, 1]
        logit = self.out_ffn_layer(rnn_seq_output[..., :, -1, :])

        # shape -> [..., n_seqs]
        logit = logit[..., 0]

        return logit


class RNN(torch.nn.Module):
    rnn_type_dict = {
        'rnn': torch.nn.RNN,
        'gru': torch.nn.GRU,
        'lstm': torch.nn.LSTM
    }

    def __init__(
            self,
            feat_dim: int,
            num_cells: int,
            num_layers: int = 1,
            dropout_rate: float = 0.1,
            cell_type: str = 'lstm',
            bidirectional: bool = False,
            bias: bool = True,
    ):
        """

        Args:
            feat_dim: int
                Dimension of input.
            num_cells: int
                Dimension of hidden state.
            num_layers: int
                Number of rnn layers.
            dropout_rate: float, default 0.1
                Dropout rate of rnn layer.
            cell_type: str, default 'lstm'
                Type of rnn cell, option in ['rnn', 'gru', 'lstm'].
            bidirectional: bool, default False.
                Identify if the rnn is bidirectional.
            bias: bool, default True.
                Identify if using bias in rnn.
        """
        super(RNN, self).__init__()
        # assignment
        if num_layers == 1 and dropout_rate > 0:
            dropout_rate = 0
        self.cell_type = cell_type.lower()
        rnn_cls = self.rnn_type_dict.get(self.cell_type)

        self.rnn_layer = rnn_cls(
            input_size=feat_dim,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
            batch_first=True,
            bias=bias
        )

    def forward(self, x, initial_state=None) -> torch.Tensor:
        """

        Args:
            x: tensor with shape [..., seq_len, feat_dim]
            initial_state: tensor or tuple of tensor
                Same like rnn, gru or lstm's state.

        Returns:
            output tensor: [..., seq_len, num_directions * num_cells]
            last state tensor: (h_n, c_n) or h_n with shape [..., num_layers * num_directions, num_cells]
        """
        if self.cell_type not in ['lstm'] and type(initial_state) in (tuple, list) and len(initial_state) == 1:
            initial_state = initial_state[0]
        self.rnn_layer.flatten_parameters()

        if len(x.size()) > 3:
            head_shape = x.shape[:-2]
            out_tuple = self.rnn_layer(x.flatten(0, -3), initial_state)
            seq_out = out_tuple[0].reshape([*head_shape, *out_tuple[0].shape[-2:]])
            return seq_out, out_tuple[1]
        else:
            return self.rnn_layer(x, initial_state)
