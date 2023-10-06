import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

__all__ = [
    "ResBlock",
    "MelResNet",
    "Stretch2d",
    "UpsampleNetwork",
    "WaveRNN",
]


class ResBlock(nn.Module):
    r"""ResNet block based on *Efficient Neural Audio Synthesis* :cite:`kalchbrenner2018efficient`.

    Args:
        n_freq: the number of bins in a spectrogram. (Default: ``128``)

    Examples
        >>> resblock = ResBlock()
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
            nn.BatchNorm1d(n_freq),
        )

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the ResBlock layer.
        Args:
            specgram (Tensor): the input sequence to the ResBlock layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_freq, n_time)
        """

        return self.resblock_model(specgram) + specgram


class MelResNet(nn.Module):
    r"""MelResNet layer uses a stack of ResBlocks on spectrogram.

    Args:
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)

    Examples
        >>> melresnet = MelResNet()
        >>> input = torch.rand(10, 128, 512)  # a random spectrogram
        >>> output = melresnet(input)  # shape: (10, 128, 508)
    """

    def __init__(
        self, n_res_block: int = 10, n_freq: int = 128, n_hidden: int = 128, n_output: int = 128, kernel_size: int = 5
    ) -> None:
        super().__init__()

        ResBlocks = [ResBlock(n_hidden) for _ in range(n_res_block)]

        self.melresnet_model = nn.Sequential(
            nn.Conv1d(in_channels=n_freq, out_channels=n_hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            *ResBlocks,
            nn.Conv1d(in_channels=n_hidden, out_channels=n_output, kernel_size=1),
        )

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the MelResNet layer.
        Args:
            specgram (Tensor): the input sequence to the MelResNet layer (n_batch, n_freq, n_time).

        Return:
            Tensor shape: (n_batch, n_output, n_time - kernel_size + 1)
        """

        return self.melresnet_model(specgram)


class Stretch2d(nn.Module):
    r"""Upscale the frequency and time dimensions of a spectrogram.

    Args:
        time_scale: the scale factor in time dimension
        freq_scale: the scale factor in frequency dimension

    Examples
        >>> stretch2d = Stretch2d(time_scale=10, freq_scale=5)

        >>> input = torch.rand(10, 100, 512)  # a random spectrogram
        >>> output = stretch2d(input)  # shape: (10, 500, 5120)
    """

    def __init__(self, time_scale: int, freq_scale: int) -> None:
        super().__init__()

        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def forward(self, specgram: Tensor) -> Tensor:
        r"""Pass the input through the Stretch2d layer.

        Args:
            specgram (Tensor): the input sequence to the Stretch2d layer (..., n_freq, n_time).

        Return:
            Tensor shape: (..., n_freq * freq_scale, n_time * time_scale)
        """

        return specgram.repeat_interleave(self.freq_scale, -2).repeat_interleave(self.time_scale, -1)


class UpsampleNetwork(nn.Module):
    r"""Upscale the dimensions of a spectrogram.

    Args:
        upsample_scales: the list of upsample scales.
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)

    Examples
        >>> upsamplenetwork = UpsampleNetwork(upsample_scales=[4, 4, 16])
        >>> input = torch.rand(10, 128, 10)  # a random spectrogram
        >>> output = upsamplenetwork(input)  # shape: (10, 128, 1536), (10, 128, 1536)
    """

    def __init__(
        self,
        upsample_scales: List[int],
        n_res_block: int = 10,
        n_freq: int = 128,
        n_hidden: int = 128,
        n_output: int = 128,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        self.total_scale: int = total_scale

        self.indent = (kernel_size - 1) // 2 * total_scale
        self.resnet = MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.resnet_stretch = Stretch2d(total_scale, 1)

        up_layers = []
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=(1, scale * 2 + 1), padding=(0, scale), bias=False
            )
            torch.nn.init.constant_(conv.weight, 1.0 / (scale * 2 + 1))
            up_layers.append(stretch)
            up_layers.append(conv)
        self.upsample_layers = nn.Sequential(*up_layers)

    def forward(self, specgram: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the UpsampleNetwork layer.

        Args:
            specgram (Tensor): the input sequence to the UpsampleNetwork layer (n_batch, n_freq, n_time)

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
        upsampling_output = upsampling_output.squeeze(1)[:, :, self.indent : -self.indent]

        return upsampling_output, resnet_output


class WaveRNN(nn.Module):
    r"""WaveRNN model from *Efficient Neural Audio Synthesis* :cite:`wavernn`
    based on the implementation from `fatchord/WaveRNN <https://github.com/fatchord/WaveRNN>`_.

    The original implementation was introduced in *Efficient Neural Audio Synthesis*
    :cite:`kalchbrenner2018efficient`. The input channels of waveform and spectrogram have to be 1.
    The product of `upsample_scales` must equal `hop_length`.

    See Also:
        * `Training example <https://github.com/pytorch/audio/tree/release/0.12/examples/pipeline_wavernn>`__
        * :class:`torchaudio.pipelines.Tacotron2TTSBundle`: TTS pipeline with pretrained model.

    Args:
        upsample_scales: the list of upsample scales.
        n_classes: the number of output classes.
        hop_length: the number of samples between the starts of consecutive frames.
        n_res_block: the number of ResBlock in stack. (Default: ``10``)
        n_rnn: the dimension of RNN layer. (Default: ``512``)
        n_fc: the dimension of fully connected layer. (Default: ``512``)
        kernel_size: the number of kernel size in the first Conv1d layer. (Default: ``5``)
        n_freq: the number of bins in a spectrogram. (Default: ``128``)
        n_hidden: the number of hidden dimensions of resblock. (Default: ``128``)
        n_output: the number of output dimensions of melresnet. (Default: ``128``)

    Example
        >>> wavernn = WaveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
        >>> waveform, sample_rate = torchaudio.load(file)
        >>> # waveform shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
        >>> specgram = MelSpectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
        >>> output = wavernn(waveform, specgram)
        >>> # output shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
    """

    def __init__(
        self,
        upsample_scales: List[int],
        n_classes: int,
        hop_length: int,
        n_res_block: int = 10,
        n_rnn: int = 512,
        n_fc: int = 512,
        kernel_size: int = 5,
        n_freq: int = 128,
        n_hidden: int = 128,
        n_output: int = 128,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self._pad = (kernel_size - 1 if kernel_size % 2 else kernel_size) // 2
        self.n_rnn = n_rnn
        self.n_aux = n_output // 4
        self.hop_length = hop_length
        self.n_classes = n_classes
        self.n_bits: int = int(math.log2(self.n_classes))

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        if total_scale != self.hop_length:
            raise ValueError(f"Expected: total_scale == hop_length, but found {total_scale} != {hop_length}")

        self.upsample = UpsampleNetwork(upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.fc = nn.Linear(n_freq + self.n_aux + 1, n_rnn)

        self.rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True)
        self.rnn2 = nn.GRU(n_rnn + self.n_aux, n_rnn, batch_first=True)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(n_rnn + self.n_aux, n_fc)
        self.fc2 = nn.Linear(n_fc + self.n_aux, n_fc)
        self.fc3 = nn.Linear(n_fc, self.n_classes)

    def forward(self, waveform: Tensor, specgram: Tensor) -> Tensor:
        r"""Pass the input through the WaveRNN model.

        Args:
            waveform: the input waveform to the WaveRNN layer (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
            specgram: the input spectrogram to the WaveRNN layer (n_batch, 1, n_freq, n_time)

        Return:
            Tensor: shape (n_batch, 1, (n_time - kernel_size + 1) * hop_length, n_classes)
        """

        if waveform.size(1) != 1:
            raise ValueError("Require the input channel of waveform is 1")
        if specgram.size(1) != 1:
            raise ValueError("Require the input channel of specgram is 1")
        # remove channel dimension until the end
        waveform, specgram = waveform.squeeze(1), specgram.squeeze(1)

        batch_size = waveform.size(0)
        h1 = torch.zeros(1, batch_size, self.n_rnn, dtype=waveform.dtype, device=waveform.device)
        h2 = torch.zeros(1, batch_size, self.n_rnn, dtype=waveform.dtype, device=waveform.device)
        # output of upsample:
        # specgram: (n_batch, n_freq, (n_time - kernel_size + 1) * total_scale)
        # aux: (n_batch, n_output, (n_time - kernel_size + 1) * total_scale)
        specgram, aux = self.upsample(specgram)
        specgram = specgram.transpose(1, 2)
        aux = aux.transpose(1, 2)

        aux_idx = [self.n_aux * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0] : aux_idx[1]]
        a2 = aux[:, :, aux_idx[1] : aux_idx[2]]
        a3 = aux[:, :, aux_idx[2] : aux_idx[3]]
        a4 = aux[:, :, aux_idx[3] : aux_idx[4]]

        x = torch.cat([waveform.unsqueeze(-1), specgram, a1], dim=-1)
        x = self.fc(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=-1)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)

        x = torch.cat([x, a4], dim=-1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        # bring back channel dimension
        return x.unsqueeze(1)

    @torch.jit.export
    def infer(self, specgram: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Inference method of WaveRNN.

        This function currently only supports multinomial sampling, which assumes the
        network is trained on cross entropy loss.

        Args:
            specgram (Tensor):
                Batch of spectrograms. Shape: `(n_batch, n_freq, n_time)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``specgram`` contains spectrograms with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The inferred waveform of size `(n_batch, 1, n_time)`.
                1 stands for a single channel.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """

        device = specgram.device
        dtype = specgram.dtype

        specgram = torch.nn.functional.pad(specgram, (self._pad, self._pad))
        specgram, aux = self.upsample(specgram)
        if lengths is not None:
            lengths = lengths * self.upsample.total_scale

        output: List[Tensor] = []
        b_size, _, seq_len = specgram.size()

        h1 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        h2 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        x = torch.zeros((b_size, 1), device=device, dtype=dtype)

        aux_split = [aux[:, self.n_aux * i : self.n_aux * (i + 1), :] for i in range(4)]

        for i in range(seq_len):

            m_t = specgram[:, :, i]

            a1_t, a2_t, a3_t, a4_t = [a[:, :, i] for a in aux_split]

            x = torch.cat([x, m_t, a1_t], dim=1)
            x = self.fc(x)
            _, h1 = self.rnn1(x.unsqueeze(1), h1)

            x = x + h1[0]
            inp = torch.cat([x, a2_t], dim=1)
            _, h2 = self.rnn2(inp.unsqueeze(1), h2)

            x = x + h2[0]
            x = torch.cat([x, a3_t], dim=1)
            x = F.relu(self.fc1(x))

            x = torch.cat([x, a4_t], dim=1)
            x = F.relu(self.fc2(x))

            logits = self.fc3(x)

            posterior = F.softmax(logits, dim=1)

            x = torch.multinomial(posterior, 1).float()
            # Transform label [0, 2 ** n_bits - 1] to waveform [-1, 1]
            x = 2 * x / (2**self.n_bits - 1.0) - 1.0

            output.append(x)

        return torch.stack(output).permute(1, 2, 0), lengths
