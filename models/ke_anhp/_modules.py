
import numpy as np
import torch


class ANHPMultiHeadAttention(torch.nn.Module):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int = None,
            dropout_rate: float = 0.4,
            n_heads: int = 4,
            activation=None,
    ):
        """

        Args:
            feat_dim: int
                Dimension of query, key and value.
            hidden_dim: int
                Dimension of the transformed query, key and value.
            dropout_rate: float
                Dropout rate of attention weights.
            n_heads: int
                Number of the attention head.
            activation: function
                Activation function of transformation layer.
        """
        super().__init__()
        # guarantee that hidden_dim is divisible by n_heads
        hidden_dim = hidden_dim or feat_dim
        if n_heads >= hidden_dim:
            n_heads = hidden_dim
        else:
            hidden_dim = hidden_dim - hidden_dim % n_heads

        # assignment
        self.n_heads = n_heads

        # layers
        self.q_trans_layer = Dense(feat_dim, hidden_dim, activation=activation)
        self.k_trans_layer = Dense(feat_dim, hidden_dim, activation=activation)
        self.v_trans_layer = Dense(feat_dim, hidden_dim, activation=activation)
        self.attention_layer = Attention(
            dropout_rate,
            mask_method='causality'
        )
        self.output_dense_layer = Dense(hidden_dim, feat_dim, activation=activation)

    def forward(
            self,
            queries,
            keys=None,
            values=None,
    ):
        """ Calculate context vector for each query.

        Args:
            queries: tensor with shape [..., n_q, feat_dim]
            keys: tensor with shape [..., n_k, feat_dim]
            values: tensor with shape [..., n_k, feat_dim]

        Returns:
            Context vector: Tensor with shape [..., n_q, hidden_dim]
        """
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        # Transformation
        # shape -> [..., hidden_dim]
        q = self.q_trans_layer(queries)
        k = self.k_trans_layer(keys)
        v = self.v_trans_layer(values)

        # Split heads
        # shape -> [n_heads * batch_size, n_q, hidden_dim // n_heads]
        q_ = self.split_head_func(q)
        # shape -> [n_heads * batch_size, n_k, hidden_dim // n_heads]
        k_ = self.split_head_func(k)
        v_ = self.split_head_func(v)

        context_vec, _ = self.attention_layer(q_, k_, v_)
        # shape -> [batch_size, n_q, hidden_dim]
        output = self.reverse_head_func(context_vec)
        return output

    def split_head_func(self, tensor):
        return torch.cat(torch.chunk(tensor, self.n_heads, dim=-1), dim=0)

    def reverse_head_func(self, tensor):
        return torch.cat(torch.chunk(tensor, self.n_heads, dim=0), dim=-1)


class Dense(torch.nn.Module):
    def __init__(
            self,
            feat_dim,
            num_cells,
            activation=None,
            bias: bool = True,
            dropout_rate: float = 0.0
    ):
        """ Fully connected layer with activation.

        Args:
            feat_dim: int
                Dimension of input tensor.
            num_cells: int
                Dimension of output tensor.
            activation: function
                Activation function.
            bias: bool
                Whether to use bias in linear layer.
            dropout_rate: float
                Rate of the dropout layer after linear production.
        """
        super(Dense, self).__init__()
        self.num_cells = num_cells
        self.activation = activation
        self.linear_layer = torch.nn.Linear(feat_dim, num_cells, bias=bias)
        self.dropout_layer = torch.nn.Dropout(dropout_rate) if dropout_rate > 0 else torch.nn.Identity()

    def forward(self, x):
        """

        Args:
            x: tensor with shape [..., feat_dim]

        Returns:
            Tensor: [..., num_cells]
        """
        x = self.linear_layer(x)
        if self.activation:
            x = self.activation(x)
        x = self.dropout_layer(x)

        return x


class Attention(torch.nn.Module):
    def __init__(
            self,
            dropout_rate: float = 0.0,
            mask_method: str = None,
            weight_norm_method='softmax'
    ):
        """ General attention layer, contains many score methods and mask methods.

        Args:
            dropout_rate: float, default 0.0
                Dropout rate of attention weights.
            mask_method: str, default None, option in [None, 'causality', 'causality-exclude']
                The method masking keys.
            weight_norm_method: str or function, option in ['softmax', 'sigmoid']
                Normalization method for attention weights.
        """
        super(Attention, self).__init__()
        self.dropout_rate = dropout_rate
        self.mask_method = mask_method

        self.att_dropout_layer = torch.nn.Dropout(self.dropout_rate)

        # shape of score -> [..., n_q, n_k]
        self.score_func = exp_scaled_dot_score_method

        # shape of attention weights -> [..., n_q, n_k]
        if mask_method is None:
            if weight_norm_method == 'sigmoid':
                self.attention_func = torch.nn.Sigmoid()
            elif weight_norm_method == 'softmax':
                self.attention_func = torch.nn.Softmax(dim=-1)
            else:
                self.attention_func = weight_norm_method
        else:
            mask_val = - 2 ** 7
            if mask_method == 'causality':
                mov_diagonal = 1
            elif mask_method == 'causality-exclude':
                mov_diagonal = 0
            else:
                raise RuntimeError('Wrong mask_method:', mask_method)

            def attention_func(score):
                if weight_norm_method == 'sigmoid':
                    att_weights = torch.sigmoid(score)
                else:
                    # shape -> [n_q, n_k]
                    up_tri_mask = torch.triu(torch.ones_like(score[0]) * mask_val, diagonal=mov_diagonal)
                    att_weights = torch.softmax(score + up_tri_mask, dim=-1)

                return att_weights

            self.attention_func = attention_func

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor = None
    ):
        """ Calculate context vector for each queries using attention.

        Args:
            queries: tensor with shape [..., n_q, feat_dim]
            keys: tensor with shape [..., n_k, feat_dim]
            values: tensor with shape [..., n_k, output_dim]

        Returns:
            Vector of weighted sums of values: Tensor with shape [..., n_q, output_dim]
            Attention Weights: Tensor with shape [..., n_q, n_k]
        """
        if queries is None:
            return values.mean(dim=-2, keepdim=True), None
        if keys is None:
            keys = queries
        if values is None:
            values = keys
        # shape -> [..., n_q, n_k]
        score = self.score_func(queries, keys)
        # shape -> [..., n_q, n_k]
        att_weights = self.attention_func(score)
        att_weights = self.att_dropout_layer(att_weights)

        # shape -> [..., n_q, n_k], [..., n_q, dim]
        context_vector = torch.matmul(att_weights, values)
        return context_vector, att_weights


def exp_scaled_dot_score_method(queries, keys):
    """ Exponential scaled dot score function.

    Args:
        queries: tensor with shape [..., n_q, feat_dim]
        keys: tensor with shape [..., n_k, feat_dim]
    Returns:
        Scores for : tensor with shape [..., n_q, n_k].
    """
    return torch.exp(
        torch.matmul(
            queries,
            keys.transpose(-2, -1)
        ) / np.sqrt(queries.size()[-1]).astype(np.float32)
    )
