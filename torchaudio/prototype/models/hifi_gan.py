"""
MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d


class HiFiGANVocoder(torch.nn.Module):
    """Generator part of *HiFi GAN* :cite:`NEURIPS2020_c5d73680`.
    Source: https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/models.py#L75

    Note:
        To build the model, please use one of the factory functions: :py:func:`hifigan_vocoder`,
        :py:func:`hifigan_vocoder_v1`, :py:func:`hifigan_vocoder_v2`, :py:func:`hifigan_vocoder_v3`.

    Args:
        in_channels (int): Number of channels in the input features.
        upsample_rates (tuple of ``int``): Factors by which each upsampling layer increases the time dimension.
        upsample_initial_channel (int): Number of channels in the input feature tensor.
        upsample_kernel_sizes (tuple of ``int``): Kernel size for each upsampling layer.
        resblock_kernel_sizes (tuple of ``int``): Kernel size for each residual block.
        resblock_dilation_sizes (tuple of tuples of ``int``): Dilation sizes for each 1D convolutional layer in each
            residual block. For resblock type 1 inner tuples should have length 3, because there are 3
            convolutions in each layer. For resblock type 2 they should have length 2.
        resblock_type (int, 1 or 2): Determines whether ``ResBlock1`` or ``ResBlock2`` will be used.
        lrelu_slope (float): Slope of leaky ReLUs in activations.
    """

    def __init__(
        self,
        in_channels: int,
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        resblock_type: int,
        lrelu_slope: float,
    ):
        super(HiFiGANVocoder, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock_type == 1 else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for (k, d) in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, d, lrelu_slope))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3)
        self.lrelu_slope = lrelu_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Feature input tensor of shape `(batch_size, num_channels, time_length)`.

        Returns:
            Tensor of shape `(batch_size, 1, time_length * upsample_rate)`, where `upsample_rate` is the product
            of upsample rates for all layers.
        """
        x = self.conv_pre(x)
        for i, upsampling_layer in enumerate(self.ups):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = upsampling_layer(x)
            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                res_block: ResBlockInterface = self.resblocks[i * self.num_kernels + j]
                xs += res_block.forward(x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


@torch.jit.interface
class ResBlockInterface(torch.nn.Module):
    """Interface for ResBlock - necessary to make type annotations in ``HiFiGANVocoder.forward`` compatible
    with TorchScript
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class ResBlock1(torch.nn.Module):
    """Residual block of type 1 for HiFiGAN Vocoder :cite:`NEURIPS2020_c5d73680`.
    Args:
        channels (int): Number of channels in the input features.
        kernel_size (int, optional): Kernel size for 1D convolutions. (Default: ``3``)
        dilation (tuple of 3 ``int``, optional): Dilations for each 1D convolution. (Default: ``(1, 3, 5)``)
        lrelu_slope (float): Slope of leaky ReLUs in activations.
    """

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int, int] = (1, 3, 5), lrelu_slope: float = 0.1
    ):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)),
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)),
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)),
            ]
        )
        self.lrelu_slope = lrelu_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input of shape ``(batch_size, channels, time_length)``.
        Returns:
            Tensor of the same shape as input.
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, self.lrelu_slope)
            xt = conv2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    """Residual block of type 2 for HiFiGAN Vocoder :cite:`NEURIPS2020_c5d73680`.
    Args:
        channels (int): Number of channels in the input features.
        kernel_size (int, optional): Kernel size for 1D convolutions. (Default: ``3``)
        dilation (tuple of 2 ``int``, optional): Dilations for each 1D convolution. (Default: ``(1, 3)``)
        lrelu_slope (float): Slope of leaky ReLUs in activations.
    """

    def __init__(
        self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int] = (1, 3), lrelu_slope: float = 0.1
    ):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
            ]
        )
        self.lrelu_slope = lrelu_slope

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (Tensor): input of shape ``(batch_size, channels, time_length)``.
        Returns:
            Tensor of the same shape as input.
        """
        for c in self.convs:
            xt = F.leaky_relu(x, self.lrelu_slope)
            xt = c(xt)
            x = xt + x
        return x


def get_padding(kernel_size, dilation=1):
    """Find padding for which 1D convolution preserves the input shape."""
    return int((kernel_size * dilation - dilation) / 2)


def hifigan_vocoder(
    in_channels: int,
    upsample_rates: Tuple[int, ...],
    upsample_initial_channel: int,
    upsample_kernel_sizes: Tuple[int, ...],
    resblock_kernel_sizes: Tuple[int, ...],
    resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
    resblock_type: int,
    lrelu_slope: float,
) -> HiFiGANVocoder:
    r"""Builds HiFi GAN Vocoder :cite:`NEURIPS2020_c5d73680`.

    Args:
        in_channels (int): See :py:class:`HiFiGANVocoder`.
        upsample_rates (tuple of ``int``): See :py:class:`HiFiGANVocoder`.
        upsample_initial_channel (int): See :py:class:`HiFiGANVocoder`.
        upsample_kernel_sizes (tuple of ``int``): See :py:class:`HiFiGANVocoder`.
        resblock_kernel_sizes (tuple of ``int``): See :py:class:`HiFiGANVocoder`.
        resblock_dilation_sizes (tuple of tuples of ``int``): See :py:class:`HiFiGANVocoder`.
        resblock_type (int, 1 or 2): See :py:class:`HiFiGANVocoder`.
    Returns:
        HiFiGANVocoder: generated model.
    """

    return HiFiGANVocoder(
        upsample_rates=upsample_rates,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        resblock_type=resblock_type,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        in_channels=in_channels,
        lrelu_slope=lrelu_slope,
    )


def hifigan_vocoder_v1() -> HiFiGANVocoder:
    r"""Builds HiFiGAN Vocoder with V1 architecture :cite:`NEURIPS2020_c5d73680`.

    Returns:
        HiFiGANVocoder: generated model.
    """
    return hifigan_vocoder(
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        resblock_type=1,
        in_channels=80,
        lrelu_slope=0.1,
    )


def hifigan_vocoder_v2() -> HiFiGANVocoder:
    r"""Builds HiFiGAN Vocoder with V2 architecture :cite:`NEURIPS2020_c5d73680`.

    Returns:
        HiFiGANVocoder: generated model.
    """
    return hifigan_vocoder(
        upsample_rates=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=128,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        resblock_type=1,
        in_channels=80,
        lrelu_slope=0.1,
    )


def hifigan_vocoder_v3() -> HiFiGANVocoder:
    r"""Builds HiFiGAN Vocoder with V3 architecture :cite:`NEURIPS2020_c5d73680`.

    Returns:
        HiFiGANVocoder: generated model.
    """
    return hifigan_vocoder(
        upsample_rates=(8, 8, 4),
        upsample_kernel_sizes=(16, 16, 8),
        upsample_initial_channel=256,
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        resblock_type=2,
        in_channels=80,
        lrelu_slope=0.1,
    )
