from typing import List

from torch import Tensor
from torch import nn

__all__ = ["_ResBlock", "_MelResNet", "_Stretch2d", "_UpsampleNetwork"]


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


class _Stretch2d(nn.Module):
    r"""Upscale the frequency and time dimensions of a spectrogram.

    Args:
        time_scale: the scale factor in time dimension
        freq_scale: the scale factor in frequency dimension

    Examples
        >>> stretch2d = _Stretch2d(time_scale=10, freq_scale=5)

        >>> input = torch.rand(10, 100, 512)  # a random spectrogram
        >>> output = stretch2d(input)  # shape: (10, 500, 5120)
    """

    def __init__(self,
                 time_scale: int,
                 freq_scale: int) -> None:
        super().__init__()

        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the _Stretch2d layer.

        Args:
            specgram (Tensor): the input sequence to the _Stretch2d layer (..., n_freq, n_time).

        Return:
            Tensor shape: (..., n_freq * freq_scale, n_time * time_scale)
        """

        return specgram.repeat_interleave(self.freq_scale, -2).repeat_interleave(self.time_scale, -1)


class _UpsampleNetwork(nn.Module):
    r"""Upscale the dimensions of a spectrogram.

    Args:
        upsample_scales: the list of upsample scales
        n_res_block: the number of ResBlock in stack (default=10)
        n_freq: the number of bins in a spectrogram (default=128)
        n_hidden: the number of hidden dimensions (default=128)
        n_output: the number of output dimensions (default=128)
        kernel_size: the number of kernel size in the first Conv1d layer (default=5)

    Examples
        >>> upsamplenetwork = _UpsampleNetwork(upsample_scales=[4, 4, 16])
        >>> input = torch.rand(10, 128, 10)  # a random spectrogram
        >>> output = upsamplenetwork(input)  # shape: (10, 1536, 128), (10, 1536, 128)
=======
    r"""This is a two-dimensional stretch layer. It is a block used in WaveRNN. WaveRNN is based on the paper
    "Efficient Neural Audio Synthesis". Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury,
    Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron van den Oord, Sander Dieleman,
    Koray Kavukcuoglu. arXiv:1802.08435, 2018.

    Args:
        x_scale: the scale factor in x axis (required).
        y_scale: the scale factor in y axis (required).

    Examples::
        >>> stretch2d = _Stretch2d(x_scale=1, y_scale=1)

        >>> input = torch.rand(10, 1, 100, 512)
        >>> output = stretch2d(input)
    """

    def __init__(self,
                 x_scale: int,
                 y_scale: int) -> None:
        super().__init__()

        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through the _Stretch2d layer.

        Args:
            x: the input sequence to the _Stretch2d layer (required).

        Shape:
            - x: :math:`(N, C, S, T)`.
            - output: :math:`(N, C, S * y_scale, T * x_scale)`.
        where N is the batch size, C is the channel size, S is the number of input sequence,
        T is the length of input sequence.
        """

        n, c, s, t = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(n, c, s * self.y_scale, t * self.x_scale)


class _UpsampleNetwork(nn.Module):
    r"""This is an upsample block based on a stack of Conv2d and Strech2d layers.
    It is a block used in WaveRNN. WaveRNN is based on the paper "Efficient Neural Audio Synthesis".
    Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart,
    Florian Stimberg, Aaron van den Oord, Sander Dieleman, Koray Kavukcuoglu. arXiv:1802.08435, 2018.

    Args:
        upsample_scales: the list of upsample scales (required).
        res_blocks: the number of ResBlock in stack (default=10).
        input_dims: the number of input sequence (default=100).
        hidden_dims: the number of compute dimensions (default=128).
        output_dims: the number of output sequence (default=128).
        pad: the number of kernal size (pad * 2 + 1) in the first Conv1d layer (default=2).

    Examples::
        >>> upsamplenetwork = _UpsampleNetwork(upsample_scales=[4, 4, 16],
                                               res_blocks=10,
                                               input_dims=100,
                                               hidden_dims=128,
                                               output_dims=128,
                                               pad=2)
        >>> input = torch.rand(10, 100, 512)
        >>> output = upsamplenetwork(input)

        super().__init__()

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale

        self.indent = (kernel_size - 1) // 2 * total_scale
        self.resnet = _MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)

        self.resnet_stretch = _Stretch2d(total_scale, 1)

        up_layers = []
        for scale in upsample_scales:
            stretch = _Stretch2d(scale, 1)
            conv = nn.Conv2d(in_channels=1,
                             out_channels=1,
                             kernel_size=(1, scale * 2 + 1),
                             padding=(0, scale),
                             bias=False)
            conv.weight.data.fill_(1. / (scale * 2 + 1))

            up_layers.append(stretch)
            up_layers.append(conv)
        self.upsample_layers = nn.Sequential(*up_layers)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the _UpsampleNetwork layer.

        Args:
            specgram (Tensor): the input sequence to the _UpsampleNetwork layer (n_batch, n_freq, n_time)

        Return:
            Tensor shape: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale),
                          (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
        where total_scale is the product of all elements in upsample_scales.
        """

        resnet_output = self.resnet(specgram).unsqueeze(1)
        resnet_output = self.resnet_stretch(resnet_output)
        resnet_output = resnet_output.squeeze(1)

        specgram = specgram.unsqueeze(1)
        upsampling_output = self.upsample_layers(specgram)
        upsampling_output = upsampling_output.squeeze(1)[:, :, self.indent:-self.indent]

        return upsampling_output, resnet_output

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through the _UpsampleNetwork layer.

        Args:
            x: the input sequence to the _UpsampleNetwork layer (required).

        Shape:
            - x: :math:`(N, S, T)`.
            - output: :math:`(N, (T - 2 * pad) * Total_Scale, S)`, `(N, (T - 2 * pad) * total_scale, P)`.
        where N is the batch size, S is the number of input sequence, T is the length of input sequence.
        P is the number of output sequence. Total_Scale is the product of all elements in upsample_scales.
        """

        resnet_output = self.resnet(x).unsqueeze(1)
        resnet_output = self.resnet_stretch(resnet_output)
        resnet_output = resnet_output.squeeze(1)

        upsampling_output = self.upsample_layers(x.unsqueeze(1))
        upsampling_output = upsampling_output.squeeze(1)[:, :, self.indent:-self.indent]

        return upsampling_output.transpose(1, 2), resnet_output.transpose(1, 2)
