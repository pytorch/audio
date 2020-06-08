from typing import Optional

from torch import Tensor
from torch import nn

__all__ = ["ResBlock", "MelResNet"]


class ResBlock(nn.Module):
    r"""
    Args:
        num_dims (int, optional): Number of compute dimensions in ResBlock. (Default: ``128``)
    """
    def __init__(self, num_dims: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(num_dims, num_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(num_dims, num_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(num_dims)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm2 = nn.BatchNorm1d(num_dims)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    r"""
    Args:
        res_blocks (int, optional): Number of ResBlocks. (Default: ``40``).
        input_dims (int, optional): Number of input dimensions (Default: ``100``).
        hidden_dims (int, optional): Number of hidden dimensions (Default: ``128``).
        output_dims (int, optional): Number of ouput dimensions (Default: ``128``).
    """
    def __init__(self, res_blocks: int = 10,
                 input_dims: int = 100,
                 hidden_dims: int = 128,
                 output_dims: int = 128) -> None:
        super().__init__()

        self.conv_in = nn.Conv1d(input_dims, hidden_dims, kernel_size=5, bias=False)
        self.batch_norm = nn.BatchNorm1d(hidden_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(hidden_dims))
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv1d(hidden_dims, output_dims, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): Tensor of dimension (batch_size, input_dims, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, output_dims, input_length-4).
        """

        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x
