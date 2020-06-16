from typing import Optional

from torch import Tensor
from torch import nn

__all__ = ["_ResBlock", "_MelResNet"]


class _ResBlock(nn.Module):
    r"""This is a ResNet block layer. This layer is based on the paper "Deep Residual Learning
    for Image Recognition". Kaiming He,  Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR, 2016.
    It is a block used in WaveRNN. WaveRNN is based on the paper "Efficient Neural Audio Synthesis".
    Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart,
    Florian Stimberg, Aaron van den Oord, Sander Dieleman, Koray Kavukcuoglu. arXiv:1802.08435, 2018.

    Args:
        num_dims: the number of compute dimensions in the input (default=128).

    Examples::
        >>> resblock = _ResBlock(num_dims=128)
        >>> input = torch.rand(10, 128, 512)
        >>> output = resblock(input)
    """

    def __init__(self, num_dims: int = 128) -> None:
        super().__init__()

        self.resblock_model = nn.Sequential(
            nn.Conv1d(in_channels=num_dims, out_channels=num_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_dims),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=num_dims, out_channels=num_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_dims)
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through the _ResBlock layer.

        Args:
            x: the input sequence to the _ResBlock layer (required).

        Shape:
            - x: :math:`(N, S, T)`.
            - output: :math:`(N, S, T)`.
        where N is the batch size, S is the number of input sequence,
        T is the length of input sequence.
        """

        residual = x
        return self.resblock_model(x) + residual


class _MelResNet(nn.Module):
    r"""This is a MelResNet layer based on a stack of ResBlocks. It is a block used in WaveRNN.
    WaveRNN is based on the paper "Efficient Neural Audio Synthesis". Nal Kalchbrenner, Erich Elsen,
    Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron van den Oord,
    Sander Dieleman, Koray Kavukcuoglu. arXiv:1802.08435, 2018.

    Args:
        res_blocks: the number of ResBlock in stack (default=10).
        input_dims: the number of input sequence (default=100).
        hidden_dims: the number of compute dimensions (default=128).
        output_dims: the number of output sequence (default=128).
        pad: the number of kernal size (pad * 2 + 1) in the first Conv1d layer (default=2).

    Examples::
        >>> melresnet = _MelResNet(res_blocks=10, input_dims=100,
                                  hidden_dims=128, output_dims=128, pad=2)
        >>> input = torch.rand(10, 100, 512)
        >>> output = melresnet(input)
    """

    def __init__(self, res_blocks: int = 10,
                 input_dims: int = 100,
                 hidden_dims: int = 128,
                 output_dims: int = 128,
                 pad: int = 2) -> None:
        super().__init__()

        kernel_size = pad * 2 + 1
        ResBlocks = []

        for i in range(res_blocks):
            ResBlocks.append(_ResBlock(hidden_dims))

        self.melresnet_model = nn.Sequential(
            nn.Conv1d(in_channels=input_dims, out_channels=hidden_dims, kernel_size=kernel_size, bias=False),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(inplace=True),
            *ResBlocks,
            nn.Conv1d(in_channels=hidden_dims, out_channels=output_dims, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through the _MelResNet layer.

        Args:
            x: the input sequence to the _MelResNet layer (required).

        Shape:
            - x: :math:`(N, S, T)`.
            - output: :math:`(N, P, T-2*pad)`.
        where N is the batch size, S is the number of input sequence,
        P is the number of ouput sequence, T is the length of input sequence.
        """

        return self.melresnet_model(x)
