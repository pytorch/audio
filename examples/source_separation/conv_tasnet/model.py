"""Implements Conv-TasNet with building blocks of it.

Based on https://github.com/naplab/Conv-TasNet/tree/e66d82a8f956a69749ec8a4ae382217faa097c5c
"""

from typing import Tuple, Optional

import torch


class ConvBlock(torch.nn.Module):
    """1D Convolutional block.

    Args:
        channels (int): The number of input/output channels, <B, Sc>
        hidden_channels (int): The number of channels in the internal layers, <H>.
        kernel_size (int): The convolution kernel size of the middle layer, <P>.
        padding (int): Padding value of the convolution in the middle layer.
        dilation (int): Dilation value of the convolution in the middle layer.
        no_redisual (bool): Disable residual block/output.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        io_channels: int,
        hidden_channels: int,
        kernel_size: int,
        padding: int,
        dilation: int = 1,
        no_residual: bool = False,
    ):
        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=io_channels, out_channels=hidden_channels, kernel_size=1
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
            torch.nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_channels,
            ),
            torch.nn.PReLU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=hidden_channels, eps=1e-08),
        )

        self.res_out = (
            None
            if no_residual
            else torch.nn.Conv1d(
                in_channels=hidden_channels, out_channels=io_channels, kernel_size=1
            )
        )
        self.skip_out = torch.nn.Conv1d(
            in_channels=hidden_channels, out_channels=io_channels, kernel_size=1
        )

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out


class MaskGenerator(torch.nn.Module):
    """TCN (Temporal Convolution Network) Separation Module

    Generates masks for separation.

    Args:
        input_dim (int): Input feature dimension, <N>.
        num_sources (int): The number of sources to separate.
        kernel_size (int): The convolution kernel size of conv blocks, <P>.
        num_featrs (int): Input/output feature dimenstion of conv blocks, <B, Sc>.
        num_hidden (int): Intermediate feature dimention of conv blocks, <H>
        num_layers (int): The number of conv blocks in one stack, <X>.
        num_stacks (int): The number of conv block stacks, <R>.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        input_dim: int,
        num_sources: int,
        kernel_size: int,
        num_feats: int,
        num_hidden: int,
        num_layers: int,
        num_stacks: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_sources = num_sources

        self.norm_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=1, num_channels=input_dim, eps=1e-8),
            torch.nn.Conv1d(
                in_channels=input_dim, out_channels=num_feats, kernel_size=1
            ),
        )
        self.receptive_field = 0
        self.conv_layers = torch.nn.ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2 ** l
                self.conv_layers.append(
                    ConvBlock(
                        io_channels=num_feats,
                        hidden_channels=num_hidden,
                        kernel_size=kernel_size,
                        dilation=multi,
                        padding=multi,
                        # The last ConvBlock does not need residual
                        no_residual=(l == (num_layers - 1) and s == (num_stacks - 1)),
                    )
                )
                self.receptive_field += (
                    kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi
                )
        self.output_layer = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Conv1d(
                in_channels=num_feats,
                out_channels=input_dim * num_sources,
                kernel_size=1,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Generate separation mask.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            torch.Tensor: shape [batch, num_sources, features, frames]
        """
        batch_size = input.shape[0]
        feats = self.norm_layers(input)
        output = 0.0
        for layer in self.conv_layers:
            residual, skip = layer(feats)
            if residual is not None:  # the last conv layer does not produce residual
                feats = feats + residual
            output = output + skip
        output = self.output_layer(output)
        return output.view(batch_size, self.num_sources, self.input_dim, -1)


class ConvTasNet(torch.nn.Module):
    """Conv-TasNet: a fully-convolutional time-domain audio separation network

    Args:
        num_sources (int): The number of sources to split.
        enc_kernel_size (int): The convolution kernel size of the encoder/decoder, <L>.
        enc_num_feats (int): The feature dimensions passed to mask generator, <N>.
        msk_kernel_size (int): The convolution kernel size of the mask generator, <P>.
        msk_num_feats (int): The input/output feature dimension of conv block in the mask generator, <B, Sc>.
        msk_num_hidden_feats (int): The internal feature dimension of conv block of the mask generator, <H>.
        msk_num_layers (int): The number of layers in one conv block of the mask generator, <X>.
        msk_num_stacks (int): The numbr of conv blocks of the mask generator, <R>.

    Note:
        This implementation corresponds to the "non-causal" setting in the paper.

    References:
        - Conv-TasNet: Surpassing Ideal Time--Frequency Magnitude Masking for Speech Separation
          Luo, Yi and Mesgarani, Nima
          https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        num_sources: int = 2,
        # encoder/decoder parameters
        enc_kernel_size: int = 16,
        enc_num_feats: int = 512,
        # mask generator parameters
        msk_kernel_size: int = 3,
        msk_num_feats: int = 128,
        msk_num_hidden_feats: int = 512,
        msk_num_layers: int = 8,
        msk_num_stacks: int = 3,
    ):
        super().__init__()

        self.num_sources = num_sources
        self.enc_num_feats = enc_num_feats
        self.enc_kernel_size = enc_kernel_size
        self.enc_stride = enc_kernel_size // 2

        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_feats,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )
        self.mask_generator = MaskGenerator(
            input_dim=enc_num_feats,
            num_sources=num_sources,
            kernel_size=msk_kernel_size,
            num_feats=msk_num_feats,
            num_hidden=msk_num_hidden_feats,
            num_layers=msk_num_layers,
            num_stacks=msk_num_stacks,
        )
        self.decoder = torch.nn.ConvTranspose1d(
            in_channels=enc_num_feats,
            out_channels=1,
            kernel_size=enc_kernel_size,
            stride=self.enc_stride,
            padding=self.enc_stride,
            bias=False,
        )

    def _pad_input(self, input: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Pad input Tensor so that the end of the input tensor corresponds with

        1. (if kernel size is odd) the center of the last convolution kernel
        or 2. (if kernel size is even) the end of the first half of the last convolution kernel

        Assuming that the resulting Tensor will be zero-padded with the size of stride
        on the both ends in Conv1D

        |<--- k_1 --->|
        |      |            |<-- k_n-1 -->|
        |      |                  |  |<--- k_n --->|
        |      |                  |         |      |
        |      |                  |         |      |
        |      v                  v         v      |
        |<---->|<--- input signal --->|<--->|<---->|
         stride                         PAD  stride

        Args:
            input (torch.Tensor): 3D Tensor with shape (batch_size, channels==1, frames)

        Returns:
            torch.Tensor: Padded Tensor
            int: Number of paddings performed
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (torch.Tensor): 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            torch.Tensor: 3D Tensor with shape [batch, channel==num_sources, frames]
        """
        if input.ndim != 3 or input.shape[1] != 1:
            raise ValueError(
                f"Expected 3D tensor (batch, channel==1, frames). Found: {input.shape}"
            )

        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources

        padded, num_pads = self._pad_input(input)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]
        feats = self.encoder(padded)  # B, F, M
        masked = self.mask_generator(feats) * feats.unsqueeze(1)  # B, S, F, M
        masked = masked.view(
            batch_size * self.num_sources, self.enc_num_feats, -1
        )  # B*S, F, M
        decoded = self.decoder(masked)  # B*S, 1, L'
        output = decoded.view(
            batch_size, self.num_sources, num_padded_frames
        )  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        return output
