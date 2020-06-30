from torch import Tensor
from torch import nn

__all__ = ["_ResBlock", "_MelResNet"]


class _ResBlock(nn.Module):
    r"""ResNet block based on "Deep Residual Learning for Image Recognition"

    The paper link is https://arxiv.org/pdf/1512.03385.pdf.

    Args:
        n_freq: the number of bins in a spectrogram (default=128)

    Examples
        >>> resblock = _ResBlock()
        >>> input = torch.rand(10, 128, 512)  # a random spectrogram
        >>> output = resblock(input)  # shape: (10, 128, 512)
    """

    def __init__(self, n_freq: int = 128) -> None:
        super().__init__()

        self.resblock_model = nn.Sequential(
            nn.Conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_freq),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False),
            nn.BatchNorm1d(n_freq)
        )

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the _ResBlock layer.
        Args:
            specgram (Tensor): the input sequence to the _ResBlock layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_freq, n_time)
        """

        return self.resblock_model(specgram) + specgram


class _MelResNet(nn.Module):
    r"""MelResNet layer uses a stack of ResBlocks on spectrogram.

    Args:
        n_res_block: the number of ResBlock in stack (default=10)
        n_freq: the number of bins in a spectrogram (default=128)
        n_hidden: the number of hidden dimensions (default=128)
        n_output: the number of output dimensions (default=128)
        kernel_size: the number of kernel size in the first Conv1d layer (default=5)

    Examples
        >>> melresnet = _MelResNet()
        >>> input = torch.rand(10, 128, 512)  # a random spectrogram
        >>> output = melresnet(input)  # shape: (10, 128, 508)
    """

    def __init__(self,
                 n_res_block: int = 10,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128,
                 kernel_size: int = 5) -> None:
        super().__init__()

        ResBlocks = [_ResBlock(n_hidden) for _ in range(n_res_block)]

        self.melresnet_model = nn.Sequential(
            nn.Conv1d(in_channels=n_freq, out_channels=n_hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            *ResBlocks,
            nn.Conv1d(in_channels=n_hidden, out_channels=n_output, kernel_size=1)
        )

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the _MelResNet layer.
        Args:
            specgram (Tensor): the input sequence to the _MelResNet layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_output, n_time - kernel_size + 1)
        """

        return self.melresnet_model(specgram)
