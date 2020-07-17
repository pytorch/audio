from typing import List

import torch
from torch import Tensor
from torch import nn

__all__ = ["_ResBlock", "_MelResNet", "_Stretch2d", "_UpsampleNetwork", "_WaveRNN"]


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
        n_hidden: the number of hidden dimensions of resblock (default=128)
        n_output: the number of output dimensions of melresnet (default=128)
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
        n_hidden: the number of hidden dimensions of resblock (default=128)
        n_output: the number of output dimensions of melresnet (default=128)
        kernel_size: the number of kernel size in the first Conv1d layer (default=5)

    Examples
        >>> upsamplenetwork = _UpsampleNetwork(upsample_scales=[4, 4, 16])
        >>> input = torch.rand(10, 128, 10)  # a random spectrogram
        >>> output = upsamplenetwork(input)  # shape: (10, 1536, 128), (10, 1536, 128)
    """

    def __init__(self,
                 upsample_scales: List[int],
                 n_res_block: int = 10,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128,
                 kernel_size: int = 5) -> None:
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


class _WaveRNN(nn.Module):
    r"""WaveRNN model based on the implementation from `fatchord <https://github.com/fatchord/WaveRNN>`_.

    The original implementation was introduced in
    `"Efficient Neural Audio Synthesis" <https://arxiv.org/pdf/1802.08435.pdf>`_.
    The input channels of waveform and spectrogram have to be 1. The product of
    `upsample_scales` must equal `hop_length`.

    Args:
        upsample_scales: the list of upsample scales
        n_classes: the number of output classes
        hop_length: the number of samples between the starts of consecutive frames
        n_res_block: the number of ResBlock in stack (default=10)
        n_rnn: the dimension of RNN layer (default=512)
        n_fc: the dimension of fully connected layer (default=512)
        kernel_size: the number of kernel size in the first Conv1d layer (default=5)
        n_freq: the number of bins in a spectrogram (default=128)
        n_hidden: the number of hidden dimensions of resblock (default=128)
        n_output: the number of output dimensions of melresnet (default=128)

    Example
        >>> wavernn = _waveRNN(upsample_scales=[5,5,8], n_classes=512, hop_length=200)
        >>> waveform, sample_rate = torchaudio.load(file)
        >>> # waveform shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length)
        >>> specgram = MelSpectrogram(sample_rate)(waveform)  # shape: (n_batch, n_channel, n_freq, n_time)
        >>> output = wavernn(waveform, specgram)
        >>> # output shape: (n_batch, n_channel, (n_time - kernel_size + 1) * hop_length, n_classes)
    """

    def __init__(self,
                 upsample_scales: List[int],
                 n_classes: int,
                 hop_length: int,
                 n_res_block: int = 10,
                 n_rnn: int = 512,
                 n_fc: int = 512,
                 kernel_size: int = 5,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.n_rnn = n_rnn
        self.n_aux = n_output // 4
        self.hop_length = hop_length
        self.n_classes = n_classes

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        if total_scale != self.hop_length:
            raise ValueError(f"Expected: total_scale == hop_length, but found {total_scale} != {hop_length}")

        self.upsample = _UpsampleNetwork(upsample_scales,
                                         n_res_block,
                                         n_freq,
                                         n_hidden,
                                         n_output,
                                         kernel_size)
        self.fc = nn.Linear(n_freq + self.n_aux + 1, n_rnn)

        self.rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True)
        self.rnn2 = nn.GRU(n_rnn + self.n_aux, n_rnn, batch_first=True)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(n_rnn + self.n_aux, n_fc)
        self.fc2 = nn.Linear(n_fc + self.n_aux, n_fc)
        self.fc3 = nn.Linear(n_fc, self.n_classes)

    def forward(self, waveform: Tensor, specgram: Tensor) -> Tensor:
        r"""Pass the input through the _WaveRNN model.

        Args:
            waveform: the input waveform to the _WaveRNN layer (n_batch, 1, (n_time - kernel_size + 1) * hop_length)
            specgram: the input spectrogram to the _WaveRNN layer (n_batch, 1, n_freq, n_time)

        Return:
            Tensor shape: (n_batch, 1, (n_time - kernel_size + 1) * hop_length, n_classes)
        """

        assert waveform.size(1) == 1, 'Require the input channel of waveform is 1'
        assert specgram.size(1) == 1, 'Require the input channel of specgram is 1'
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
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

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
