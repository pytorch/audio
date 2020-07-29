from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["_ConvNorm", "_Encoder"]


class _ConvNorm(nn.Module):
    r"""1-d convolution layer

    Args:
        n_input: the number of input channels.
        n_output: the number of output channels.

    Examples
        >>> convnorm = _ConvNorm(10, 20)
        >>> input = torch.rand(32, 10, 512)
        >>> output = convnorm(input)  # shape: (32, 20, 512)
    """

    def __init__(
        self,
        n_input,
        n_output,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super(_ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            n_input,
            n_output,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through the _ConvNorm layer.

        Args:
            x (Tensor): the input sequence to the _ConvNorm layer (n_batch, n_input, n_seq).

        Return:
            Tensor shape: (n_batch, n_output, n_seq)
        """

        return self.conv(x)


class _Encoder(nn.Module):
    r"""Encoder Module

        Args:
            n_encoder_convolutions: the number of convolution layers in the encoder.
            n_encoder_embedding: the number of embedding dimensions in the encoder.
            n_encoder_kernel_size: the kernel size in the encoder.

        Examples
            >>> encoder = _Encoder(3, 512, 5)
            >>> input = torch.rand(10, 20, 30)
            >>> output = encoder(input)  # shape: (10, 30, 512)
    """

    def __init__(
        self, n_encoder_convolutions, n_encoder_embedding, n_encoder_kernel_size
    ) -> None:
        super(_Encoder, self).__init__()

        convolutions = []
        for _ in range(n_encoder_convolutions):
            conv_layer = nn.Sequential(
                _ConvNorm(
                    n_encoder_embedding,
                    n_encoder_embedding,
                    kernel_size=n_encoder_kernel_size,
                    stride=1,
                    padding=int((n_encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(n_encoder_embedding),
            )
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            n_encoder_embedding,
            int(n_encoder_embedding / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        r"""Pass the input through the _Encoder layer.

        Args:
            x (Tensor): the input sequence to the _Encoder layer (n_batch, n_encoder_embedding, n_seq).
            input_lengths (Tensor): the length of input sequence to the _Encoder layer (n_batch,).

        Return:
            Tensor shape: (n_batch, n_seq, n_encoder_embedding)
        """

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

    def infer(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        r"""Pass the input through the _Encoder layer for inference.

        Args:
            x (Tensor): the input sequence to the _Encoder layer (n_batch, n_encoder_embedding, n_seq).
            input_lengths (Tensor): the length of input sequence to the _Encoder layer (n_batch,).

        Return:
            Tensor shape: (n_batch, n_seq, n_encoder_embedding)
        """

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs
