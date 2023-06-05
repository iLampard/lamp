
import torch


class ParameterDot(torch.nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            in1_features: int,
            in2_features: int,
            bias: bool = False,
    ):
        super(ParameterDot, self).__init__()
        self.use_bias = bias

        self.weights = torch.nn.Parameter(torch.Tensor(num_embeddings, in1_features, in2_features))
        torch.nn.init.xavier_uniform_(self.weights, gain=torch.nn.init.calculate_gain('relu'))
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty((num_embeddings,)), requires_grad=True)
            torch.nn.init.constant_(self.bias, 0.1)

    def forward(self, tensor1, tensor2, parameters_index):
        """Applies a dot transformation to the incoming data: `y = x_1^T A x_2 + b`.

        Args:
            tensor1: tensor with shape [B, ..., in1_features]
            tensor2: tensor with shape [B, ..., in2_features]
            parameters_index: tensor with shape [B]

        Returns:
            Tensor with shape [B].
        """
        # [B, in1_features, in2_features]
        weight = self.weights[parameters_index]

        # [..., B, in1_features]
        tensor1 = tensor1.transpose(-2, 0)
        # [B, ..., in2_features]
        x = torch.sum(tensor1[..., None] * weight, dim=-2).transpose(0, -2)
        # [B, ...]
        x = torch.sum(x * tensor2, dim=-1)

        if self.use_bias:
            # [B]
            bias = self.bias[parameters_index]

            x = x.transpose(-1, 0) + bias
            x = x.transpose(0, -1)

        return x
