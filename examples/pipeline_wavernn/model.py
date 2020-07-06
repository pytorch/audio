from typing import List

import torch
from torch import Tensor
from torch import nn

__all__ = ["_ResBlock", "_MelResNet", "_Stretch2d", "_UpsampleNetwork", "_WaveRNN"]


class _ResBlock(nn.Module):
    r"""This is a ResNet block layer. This layer is based on the paper "Deep Residual Learning
    for Image Recognition". Kaiming He,  Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR, 2016.
    It is a block used in WaveRNN.

    Args:
        n_freq: the number of bins in a spectrogram (default=128)

    Examples::
        >>> resblock = _ResBlock(n_freq=128)
        >>> input = torch.rand(10, 128, 512)
        >>> output = resblock(input)
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

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: the input sequence to the _ResBlock layer

        Shape:
            - x: :math:`(batch, freq, time)`
            - output: :math:`(batch, freq, time)`
        """

        residual = x
        return self.resblock_model(x) + residual


class _MelResNet(nn.Module):
    r"""This is a MelResNet layer based on a stack of ResBlocks. It is a block used in WaveRNN.

    Args:
        n_res_block: the number of ResBlock in stack (default=10)
        n_freq: the number of bins in a spectrogram (default=128)
        n_hidden: the number of hidden dimensions (default=128)
        n_output: the number of output dimensions (default=128)
        kernel_size: the number of kernel size in the first Conv1d layer (default=5)

    Examples::
        >>> melresnet = _MelResNet(n_res_block=10, n_freq=128, n_hidden=128,
                                   n_output=128, kernel_size=5)
        >>> input = torch.rand(10, 128, 512)
        >>> output = melresnet(input)
    """

    def __init__(self,
                 n_res_block: int = 10,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128,
                 kernel_size: int = 5) -> None:
        super().__init__()

        ResBlocks = []

        for i in range(n_res_block):
            ResBlocks.append(_ResBlock(n_hidden))

        self.melresnet_model = nn.Sequential(
            nn.Conv1d(in_channels=n_freq, out_channels=n_hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            *ResBlocks,
            nn.Conv1d(in_channels=n_hidden, out_channels=n_output, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: the input sequence to the _MelResNet layer

        Shape:
            - x: :math:`(batch, freq, time)`
            - output: :math:`(batch, n_output, time - kernel_size + 1)`
        """

        return self.melresnet_model(x)


class _Stretch2d(nn.Module):
    r"""This is a two-dimensional stretch layer. It is a block used in WaveRNN.

    Args:
        x_scale: the scale factor in x axis
        y_scale: the scale factor in y axis

    Examples::
        >>> stretch2d = _Stretch2d(x_scale=10, y_scale=10)

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
        r"""
        Args:
            x: the input sequence to the _Stretch2d layer

        Shape:
            - x: :math:`(..., freq, time)`
            - output: :math:`(..., freq * y_scale, time * x_scale)`
        """

        return x.repeat_interleave(self.y_scale, 2).repeat_interleave(self.x_scale, 3)


class _UpsampleNetwork(nn.Module):
    r"""This is an upsample block based on a stack of Conv2d and Strech2d layers.
    It is a block used in WaveRNN.

    Args:
        upsample_scales: the list of upsample scales
        n_res_block: the number of ResBlock in stack (default=10)
        n_freq: the number of bins in a spectrogram (default=128)
        n_hidden: the number of hidden dimensions (default=128)
        n_output: the number of output dimensions (default=128)
        kernel_size: the number of kernel size in the first Conv1d layer (default=5)

    Examples::
        >>> upsamplenetwork = _UpsampleNetwork(upsample_scales=[4, 4, 16],
                                               n_res_block=10,
                                               n_freq=128,
                                               n_hidden=128,
                                               n_output=128,
                                               kernel_size=5)
        >>> input = torch.rand(10, 128, 512)
        >>> output = upsamplenetwork(input)
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
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = _Stretch2d(scale, 1)
            conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            up_layers.append(stretch)
            up_layers.append(conv)
        self.upsample_layers = nn.Sequential(*up_layers)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x: the input sequence to the _UpsampleNetwork layer

        Shape:
            - x: :math:`(batch, freq, time)`.
            - output: :math:`(batch, (time - kernel_size + 1) * total_scale, freq)`,
                            `(batch, (time - kernel_size + 1) * total_scale, n_output)`
        where total_scale is the product of all elements in upsample_scales.
        """

        resnet_output = self.resnet(x).unsqueeze(1)
        resnet_output = self.resnet_stretch(resnet_output)
        resnet_output = resnet_output.squeeze(1)

        x = x.unsqueeze(1)
        upsampling_output = self.upsample_layers(x)
        upsampling_output = upsampling_output.squeeze(1)[:, :, self.indent:-self.indent]

        return upsampling_output.transpose(1, 2), resnet_output.transpose(1, 2)


class _WaveRNN(nn.Module):
    r"""
    Args:
        upsample_scales: the list of upsample scales
        n_bits: the bits of output waveform
        sample_rate: the rate of audio dimensions (samples per second)
        hop_length: the number of samples between the starts of consecutive frames
        n_res_block: the number of ResBlock in stack (default=10)
        n_rnn: the dimension of RNN layer (default=512)
        n_fc: the dimension of fully connected layer (default=512)
        kernel_size: the number of kernel size in the first Conv1d layer (default=5)
        n_freq: the number of bins in a spectrogram (default=128)
        n_hidden: the number of hidden dimensions (default=128)
        n_output: the number of output dimensions (default=128)
        mode: the type of input waveform (default='RAW')

    Examples::
        >>> upsamplenetwork = _waveRNN(upsample_scales=[5,5,8],
                                       n_bits=9,
                                       sample_rate=24000,
                                       hop_length=200,
                                       n_res_block=10,
                                       n_rnn=512,
                                       n_fc=512,
                                       kernel_size=5,
                                       n_freq=128,
                                       n_hidden=128,
                                       n_output=128,
                                       mode='RAW')
        >>> x = torch.rand(10, 24800, 512)
        >>> mels = torch.rand(10, 128, 512)
        >>> output = upsamplenetwork(x, mels)
    """

    def __init__(self,
                 upsample_scales: List[int],
                 n_bits: int,
                 sample_rate: int,
                 hop_length: int,
                 n_res_block: int = 10,
                 n_rnn: int = 512,
                 n_fc: int = 512,
                 kernel_size: int = 5,
                 n_freq: int = 128,
                 n_hidden: int = 128,
                 n_output: int = 128,
                 mode: str = 'waveform') -> None:
        super().__init__()

        self.mode = mode
        self.kernel_size = kernel_size

        if self.mode == 'waveform':
            self.n_classes = 2 ** n_bits
        elif self.mode == 'mol':
            self.n_classes = 30

        self.n_rnn = n_rnn
        self.n_aux = n_output // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = _UpsampleNetwork(upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.fc = nn.Linear(n_freq + self.n_aux + 1, n_rnn)

        self.rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True)
        self.rnn2 = nn.GRU(n_rnn + self.n_aux, n_rnn, batch_first=True)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(n_rnn + self.n_aux, n_fc)
        self.fc2 = nn.Linear(n_fc + self.n_aux, n_fc)
        self.fc3 = nn.Linear(n_fc, self.n_classes)

    def forward(self, x: Tensor, mels: Tensor) -> Tensor:
        r"""
        Args:
            x: the input waveform to the _WaveRNN layer
            mels: the input mel-spectrogram to the _WaveRNN layer

        Shape:
            - x: :math:`(batch, time)`
            - mels: :math:`(batch, freq, time_mels)`
            - output: :math:`(batch, time, 2 ** n_bits)`
        """

        batch_size = x.size(0)
        h1 = torch.zeros(1, batch_size, self.n_rnn, device=x.device)
        h2 = torch.zeros(1, batch_size, self.n_rnn, device=x.device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.n_aux * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.fc(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = self.relu1(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = self.relu2(self.fc2(x))

        return self.fc3(x)
