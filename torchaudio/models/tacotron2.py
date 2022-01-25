# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import warnings
from typing import Tuple, List, Optional, Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


__all__ = [
    "Tacotron2",
]


def _get_linear_layer(in_dim: int, out_dim: int, bias: bool = True, w_init_gain: str = "linear") -> torch.nn.Linear:
    r"""Linear layer with xavier uniform initialization.

    Args:
        in_dim (int): Size of each input sample.
        out_dim (int): Size of each output sample.
        bias (bool, optional): If set to ``False``, the layer will not learn an additive bias. (Default: ``True``)
        w_init_gain (str, optional): Parameter passed to ``torch.nn.init.calculate_gain``
            for setting the gain parameter of ``xavier_uniform_``. (Default: ``linear``)

    Returns:
        (torch.nn.Linear): The corresponding linear layer.
    """
    linear = torch.nn.Linear(in_dim, out_dim, bias=bias)
    torch.nn.init.xavier_uniform_(linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    return linear


def _get_conv1d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    stride: int = 1,
    padding: Optional[Union[str, int, Tuple[int]]] = None,
    dilation: int = 1,
    bias: bool = True,
    w_init_gain: str = "linear",
) -> torch.nn.Conv1d:
    r"""1D convolution with xavier uniform initialization.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int, optional): Number of channels in the input image. (Default: ``1``)
        stride (int, optional): Number of channels in the input image. (Default: ``1``)
        padding (str, int or tuple, optional): Padding added to both sides of the input.
            (Default: dilation * (kernel_size - 1) / 2)
        dilation (int, optional): Number of channels in the input image. (Default: ``1``)
        w_init_gain (str, optional): Parameter passed to ``torch.nn.init.calculate_gain``
            for setting the gain parameter of ``xavier_uniform_``. (Default: ``linear``)

    Returns:
        (torch.nn.Conv1d): The corresponding Conv1D layer.
    """
    if padding is None:
        assert kernel_size % 2 == 1
        padding = int(dilation * (kernel_size - 1) / 2)

    conv1d = torch.nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )

    torch.nn.init.xavier_uniform_(conv1d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    return conv1d


def _get_mask_from_lengths(lengths: Tensor) -> Tensor:
    r"""Returns a binary mask based on ``lengths``. The ``i``-th row and ``j``-th column of the mask
    is ``1`` if ``j`` is smaller than ``i``-th element of ``lengths.

    Args:
        lengths (Tensor): The length of each element in the batch, with shape (n_batch, ).

    Returns:
        mask (Tensor): The binary mask, with shape (n_batch, max of ``lengths``).
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = torch.le(mask, 0)
    return mask


class _LocationLayer(nn.Module):
    r"""Location layer used in the Attention model.

    Args:
        attention_n_filter (int): Number of filters for attention model.
        attention_kernel_size (int): Kernel size for attention model.
        attention_hidden_dim (int): Dimension of attention hidden representation.
    """

    def __init__(
        self,
        attention_n_filter: int,
        attention_kernel_size: int,
        attention_hidden_dim: int,
    ):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = _get_conv1d_layer(
            2,
            attention_n_filter,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = _get_linear_layer(
            attention_n_filter, attention_hidden_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        r"""Location layer used in the Attention model.

        Args:
            attention_weights_cat (Tensor): Cumulative and previous attention weights
                with shape (n_batch, 2, max of ``text_lengths``).

        Returns:
            processed_attention (Tensor): Cumulative and previous attention weights
                with shape (n_batch, ``attention_hidden_dim``).
        """
        # (n_batch, attention_n_filter, text_lengths.max())
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        # (n_batch, text_lengths.max(), attention_hidden_dim)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class _Attention(nn.Module):
    r"""Locally sensitive attention model.

    Args:
        attention_rnn_dim (int): Number of hidden units for RNN.
        encoder_embedding_dim (int): Number of embedding dimensions in the Encoder.
        attention_hidden_dim (int): Dimension of attention hidden representation.
        attention_location_n_filter (int): Number of filters for Attention model.
        attention_location_kernel_size (int): Kernel size for Attention model.
    """

    def __init__(
        self,
        attention_rnn_dim: int,
        encoder_embedding_dim: int,
        attention_hidden_dim: int,
        attention_location_n_filter: int,
        attention_location_kernel_size: int,
    ) -> None:
        super().__init__()
        self.query_layer = _get_linear_layer(attention_rnn_dim, attention_hidden_dim, bias=False, w_init_gain="tanh")
        self.memory_layer = _get_linear_layer(
            encoder_embedding_dim, attention_hidden_dim, bias=False, w_init_gain="tanh"
        )
        self.v = _get_linear_layer(attention_hidden_dim, 1, bias=False)
        self.location_layer = _LocationLayer(
            attention_location_n_filter,
            attention_location_kernel_size,
            attention_hidden_dim,
        )
        self.score_mask_value = -float("inf")

    def _get_alignment_energies(self, query: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor) -> Tensor:
        r"""Get the alignment vector.

        Args:
            query (Tensor): Decoder output with shape (n_batch, n_mels * n_frames_per_step).
            processed_memory (Tensor): Processed Encoder outputs
                with shape (n_batch, max of ``text_lengths``, attention_hidden_dim).
            attention_weights_cat (Tensor): Cumulative and previous attention weights
                with shape (n_batch, 2, max of ``text_lengths``).

        Returns:
            alignment (Tensor): attention weights, it is a tensor with shape (batch, max of ``text_lengths``).
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))

        alignment = energies.squeeze(2)
        return alignment

    def forward(
        self,
        attention_hidden_state: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        attention_weights_cat: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the Attention model.

        Args:
            attention_hidden_state (Tensor): Attention rnn last output with shape (n_batch, ``attention_rnn_dim``).
            memory (Tensor): Encoder outputs with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            processed_memory (Tensor): Processed Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``attention_hidden_dim``).
            attention_weights_cat (Tensor): Previous and cumulative attention weights
                with shape (n_batch, current_num_frames * 2, max of ``text_lengths``).
            mask (Tensor): Binary mask for padded data with shape (n_batch, current_num_frames).

        Returns:
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
        """
        alignment = self._get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class _Prenet(nn.Module):
    r"""Prenet Module. It is consists of ``len(output_size)`` linear layers.

    Args:
        in_dim (int): The size of each input sample.
        output_sizes (list): The output dimension of each linear layers.
    """

    def __init__(self, in_dim: int, out_sizes: List[int]) -> None:
        super().__init__()
        in_sizes = [in_dim] + out_sizes[:-1]
        self.layers = nn.ModuleList(
            [_get_linear_layer(in_size, out_size, bias=False) for (in_size, out_size) in zip(in_sizes, out_sizes)]
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through Prenet.

        Args:
            x (Tensor): The input sequence to Prenet with shape (n_batch, in_dim).

        Return:
            x (Tensor): Tensor with shape (n_batch, sizes[-1])
        """

        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class _Postnet(nn.Module):
    r"""Postnet Module.

    Args:
        n_mels (int): Number of mel bins.
        postnet_embedding_dim (int): Postnet embedding dimension.
        postnet_kernel_size (int): Postnet kernel size.
        postnet_n_convolution (int): Number of postnet convolutions.
    """

    def __init__(
        self,
        n_mels: int,
        postnet_embedding_dim: int,
        postnet_kernel_size: int,
        postnet_n_convolution: int,
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()

        for i in range(postnet_n_convolution):
            in_channels = n_mels if i == 0 else postnet_embedding_dim
            out_channels = n_mels if i == (postnet_n_convolution - 1) else postnet_embedding_dim
            init_gain = "linear" if i == (postnet_n_convolution - 1) else "tanh"
            num_features = n_mels if i == (postnet_n_convolution - 1) else postnet_embedding_dim
            self.convolutions.append(
                nn.Sequential(
                    _get_conv1d_layer(
                        in_channels,
                        out_channels,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain=init_gain,
                    ),
                    nn.BatchNorm1d(num_features),
                )
            )

        self.n_convs = len(self.convolutions)

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through Postnet.

        Args:
            x (Tensor): The input sequence with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).

        Return:
            x (Tensor): Tensor with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
        """

        for i, conv in enumerate(self.convolutions):
            if i < self.n_convs - 1:
                x = F.dropout(torch.tanh(conv(x)), 0.5, training=self.training)
            else:
                x = F.dropout(conv(x), 0.5, training=self.training)

        return x


class _Encoder(nn.Module):
    r"""Encoder Module.

    Args:
        encoder_embedding_dim (int): Number of embedding dimensions in the encoder.
        encoder_n_convolution (int): Number of convolution layers in the encoder.
        encoder_kernel_size (int): The kernel size in the encoder.

    Examples
        >>> encoder = _Encoder(3, 512, 5)
        >>> input = torch.rand(10, 20, 30)
        >>> output = encoder(input)  # shape: (10, 30, 512)
    """

    def __init__(
        self,
        encoder_embedding_dim: int,
        encoder_n_convolution: int,
        encoder_kernel_size: int,
    ) -> None:
        super().__init__()

        self.convolutions = nn.ModuleList()
        for _ in range(encoder_n_convolution):
            conv_layer = nn.Sequential(
                _get_conv1d_layer(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(encoder_embedding_dim),
            )
            self.convolutions.append(conv_layer)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm.flatten_parameters()

    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        r"""Pass the input through the Encoder.

        Args:
            x (Tensor): The input sequences with shape (n_batch, encoder_embedding_dim, n_seq).
            input_lengths (Tensor): The length of each input sequence with shape (n_batch, ).

        Return:
            x (Tensor): A tensor with shape (n_batch, n_seq, encoder_embedding_dim).
        """

        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        input_lengths = input_lengths.cpu()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs


class _Decoder(nn.Module):
    r"""Decoder with Attention model.

    Args:
        n_mels (int): number of mel bins
        n_frames_per_step (int): number of frames processed per step, only 1 is supported
        encoder_embedding_dim (int): the number of embedding dimensions in the encoder.
        decoder_rnn_dim (int): number of units in decoder LSTM
        decoder_max_step (int): maximum number of output mel spectrograms
        decoder_dropout (float): dropout probability for decoder LSTM
        decoder_early_stopping (bool): stop decoding when all samples are finished
        attention_rnn_dim (int): number of units in attention LSTM
        attention_hidden_dim (int): dimension of attention hidden representation
        attention_location_n_filter (int): number of filters for attention model
        attention_location_kernel_size (int): kernel size for attention model
        attention_dropout (float): dropout probability for attention LSTM
        prenet_dim (int): number of ReLU units in prenet layers
        gate_threshold (float): probability threshold for stop token
    """

    def __init__(
        self,
        n_mels: int,
        n_frames_per_step: int,
        encoder_embedding_dim: int,
        decoder_rnn_dim: int,
        decoder_max_step: int,
        decoder_dropout: float,
        decoder_early_stopping: bool,
        attention_rnn_dim: int,
        attention_hidden_dim: int,
        attention_location_n_filter: int,
        attention_location_kernel_size: int,
        attention_dropout: float,
        prenet_dim: int,
        gate_threshold: float,
    ) -> None:

        super().__init__()
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.decoder_max_step = decoder_max_step
        self.gate_threshold = gate_threshold
        self.attention_dropout = attention_dropout
        self.decoder_dropout = decoder_dropout
        self.decoder_early_stopping = decoder_early_stopping

        self.prenet = _Prenet(n_mels * n_frames_per_step, [prenet_dim, prenet_dim])

        self.attention_rnn = nn.LSTMCell(prenet_dim + encoder_embedding_dim, attention_rnn_dim)

        self.attention_layer = _Attention(
            attention_rnn_dim,
            encoder_embedding_dim,
            attention_hidden_dim,
            attention_location_n_filter,
            attention_location_kernel_size,
        )

        self.decoder_rnn = nn.LSTMCell(attention_rnn_dim + encoder_embedding_dim, decoder_rnn_dim, True)

        self.linear_projection = _get_linear_layer(decoder_rnn_dim + encoder_embedding_dim, n_mels * n_frames_per_step)

        self.gate_layer = _get_linear_layer(
            decoder_rnn_dim + encoder_embedding_dim, 1, bias=True, w_init_gain="sigmoid"
        )

    def _get_initial_frame(self, memory: Tensor) -> Tensor:
        r"""Gets all zeros frames to use as the first decoder input.

        Args:
            memory (Tensor): Encoder outputs with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).

        Returns:
            decoder_input (Tensor): all zeros frames with shape
                (n_batch, max of ``text_lengths``, ``n_mels * n_frames_per_step``).
        """

        n_batch = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(n_batch, self.n_mels * self.n_frames_per_step, dtype=dtype, device=device)
        return decoder_input

    def _initialize_decoder_states(
        self, memory: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory.

        Args:
            memory (Tensor): Encoder outputs with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).

        Returns:
            attention_hidden (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            attention_cell (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            decoder_hidden (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            decoder_cell (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
            attention_weights_cum (Tensor): Cumulated attention weights with shape (n_batch, max of ``text_lengths``).
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
            processed_memory (Tensor): Processed encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``attention_hidden_dim``).
        """
        n_batch = memory.size(0)
        max_time = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(n_batch, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(n_batch, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(n_batch, self.decoder_rnn_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(n_batch, self.decoder_rnn_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(n_batch, max_time, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(n_batch, max_time, dtype=dtype, device=device)
        attention_context = torch.zeros(n_batch, self.encoder_embedding_dim, dtype=dtype, device=device)

        processed_memory = self.attention_layer.memory_layer(memory)

        return (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        )

    def _parse_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        r"""Prepares decoder inputs.

        Args:
            decoder_inputs (Tensor): Inputs used for teacher-forced training, i.e. mel-specs,
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``)

        Returns:
            inputs (Tensor): Processed decoder inputs with shape (max of ``mel_specgram_lengths``, n_batch, ``n_mels``).
        """
        # (n_batch, n_mels, mel_specgram_lengths.max()) -> (n_batch, mel_specgram_lengths.max(), n_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (n_batch, mel_specgram_lengths.max(), n_mels) -> (mel_specgram_lengths.max(), n_batch, n_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def _parse_decoder_outputs(
        self, mel_specgram: Tensor, gate_outputs: Tensor, alignments: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Prepares decoder outputs for output

        Args:
            mel_specgram (Tensor): mel spectrogram with shape (max of ``mel_specgram_lengths``, n_batch, ``n_mels``)
            gate_outputs (Tensor): predicted stop token with shape (max of ``mel_specgram_lengths``, n_batch)
            alignments (Tensor): sequence of attention weights from the decoder
                with shape (max of ``mel_specgram_lengths``, n_batch, max of ``text_lengths``)

        Returns:
            mel_specgram (Tensor): mel spectrogram with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``)
            gate_outputs (Tensor): predicted stop token with shape (n_batch, max of ``mel_specgram_lengths``)
            alignments (Tensor): sequence of attention weights from the decoder
                with shape (n_batch, max of ``mel_specgram_lengths``, max of ``text_lengths``)
        """
        # (mel_specgram_lengths.max(), n_batch, text_lengths.max())
        # -> (n_batch, mel_specgram_lengths.max(), text_lengths.max())
        alignments = alignments.transpose(0, 1).contiguous()
        # (mel_specgram_lengths.max(), n_batch) -> (n_batch, mel_specgram_lengths.max())
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (mel_specgram_lengths.max(), n_batch, n_mels) -> (n_batch, mel_specgram_lengths.max(), n_mels)
        mel_specgram = mel_specgram.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_specgram.shape[0], -1, self.n_mels)
        mel_specgram = mel_specgram.view(*shape)
        # (n_batch, mel_specgram_lengths.max(), n_mels) -> (n_batch, n_mels, T_out)
        mel_specgram = mel_specgram.transpose(1, 2)

        return mel_specgram, gate_outputs, alignments

    def decode(
        self,
        decoder_input: Tensor,
        attention_hidden: Tensor,
        attention_cell: Tensor,
        decoder_hidden: Tensor,
        decoder_cell: Tensor,
        attention_weights: Tensor,
        attention_weights_cum: Tensor,
        attention_context: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Decoder step using stored states, attention and memory

        Args:
            decoder_input (Tensor): Output of the Prenet with shape (n_batch, ``prenet_dim``).
            attention_hidden (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            attention_cell (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            decoder_hidden (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            decoder_cell (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
            attention_weights_cum (Tensor): Cumulated attention weights with shape (n_batch, max of ``text_lengths``).
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
            memory (Tensor): Encoder output with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            processed_memory (Tensor): Processed Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``attention_hidden_dim``).
            mask (Tensor): Binary mask for padded data with shape (n_batch, current_num_frames).

        Returns:
            decoder_output: Predicted mel spectrogram for the current frame with shape (n_batch, ``n_mels``).
            gate_prediction (Tensor): Prediction of the stop token with shape (n_batch, ``1``).
            attention_hidden (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            attention_cell (Tensor): Hidden state of the attention LSTM with shape (n_batch, ``attention_rnn_dim``).
            decoder_hidden (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            decoder_cell (Tensor): Hidden state of the decoder LSTM with shape (n_batch, ``decoder_rnn_dim``).
            attention_weights (Tensor): Attention weights with shape (n_batch, max of ``text_lengths``).
            attention_weights_cum (Tensor): Cumulated attention weights with shape (n_batch, max of ``text_lengths``).
            attention_context (Tensor): Context vector with shape (n_batch, ``encoder_embedding_dim``).
        """
        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))
        attention_hidden = F.dropout(attention_hidden, self.attention_dropout, self.training)

        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory, attention_weights_cat, mask
        )

        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (
            decoder_output,
            gate_prediction,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
        )

    def forward(
        self, memory: Tensor, mel_specgram_truth: Tensor, memory_lengths: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Decoder forward pass for training.

        Args:
            memory (Tensor): Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            mel_specgram_truth (Tensor): Decoder ground-truth mel-specs for teacher forcing
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            memory_lengths (Tensor): Encoder output lengths for attention masking
                (the same as ``text_lengths``) with shape (n_batch, ).

        Returns:
            mel_specgram (Tensor): Predicted mel spectrogram
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            gate_outputs (Tensor): Predicted stop token for each timestep
                with shape (n_batch,  max of ``mel_specgram_lengths``).
            alignments (Tensor): Sequence of attention weights from the decoder
                with shape (n_batch,  max of ``mel_specgram_lengths``, max of ``text_lengths``).
        """

        decoder_input = self._get_initial_frame(memory).unsqueeze(0)
        decoder_inputs = self._parse_decoder_inputs(mel_specgram_truth)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        mask = _get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self._initialize_decoder_states(memory)

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            (
                mel_output,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_specgram, gate_outputs, alignments = self._parse_decoder_outputs(
            torch.stack(mel_outputs), torch.stack(gate_outputs), torch.stack(alignments)
        )

        return mel_specgram, gate_outputs, alignments

    def _get_go_frame(self, memory: Tensor) -> Tensor:
        """Gets all zeros frames to use as the first decoder input

        args:
            memory (Tensor): Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).

        returns:
            decoder_input (Tensor): All zeros frames with shape(n_batch, ``n_mels`` * ``n_frame_per_step``).
        """

        n_batch = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(n_batch, self.n_mels * self.n_frames_per_step, dtype=dtype, device=device)
        return decoder_input

    @torch.jit.export
    def infer(self, memory: Tensor, memory_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Decoder inference

        Args:
            memory (Tensor): Encoder outputs
                with shape (n_batch, max of ``text_lengths``, ``encoder_embedding_dim``).
            memory_lengths (Tensor): Encoder output lengths for attention masking
                (the same as ``text_lengths``) with shape (n_batch, ).

        Returns:
            mel_specgram (Tensor): Predicted mel spectrogram
                with shape (n_batch, ``n_mels``, max of ``mel_specgram_lengths``).
            mel_specgram_lengths (Tensor): the length of the predicted mel spectrogram (n_batch, ))
            gate_outputs (Tensor): Predicted stop token for each timestep
                with shape (n_batch,  max of ``mel_specgram_lengths``).
            alignments (Tensor): Sequence of attention weights from the decoder
                with shape (n_batch,  max of ``mel_specgram_lengths``, max of ``text_lengths``).
        """
        batch_size, device = memory.size(0), memory.device

        decoder_input = self._get_go_frame(memory)

        mask = _get_mask_from_lengths(memory_lengths)
        (
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context,
            processed_memory,
        ) = self._initialize_decoder_states(memory)

        mel_specgram_lengths = torch.zeros([batch_size], dtype=torch.int32, device=device)
        finished = torch.zeros([batch_size], dtype=torch.bool, device=device)
        mel_specgrams: List[Tensor] = []
        gate_outputs: List[Tensor] = []
        alignments: List[Tensor] = []
        for _ in range(self.decoder_max_step):
            decoder_input = self.prenet(decoder_input)
            (
                mel_specgram,
                gate_output,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
            ) = self.decode(
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask,
            )

            mel_specgrams.append(mel_specgram.unsqueeze(0))
            gate_outputs.append(gate_output.transpose(0, 1))
            alignments.append(attention_weights)
            mel_specgram_lengths[~finished] += 1

            finished |= torch.sigmoid(gate_output.squeeze(1)) > self.gate_threshold
            if self.decoder_early_stopping and torch.all(finished):
                break

            decoder_input = mel_specgram

        if len(mel_specgrams) == self.decoder_max_step:
            warnings.warn(
                "Reached max decoder steps. The generated spectrogram might not cover " "the whole transcript."
            )

        mel_specgrams = torch.cat(mel_specgrams, dim=0)
        gate_outputs = torch.cat(gate_outputs, dim=0)
        alignments = torch.cat(alignments, dim=0)

        mel_specgrams, gate_outputs, alignments = self._parse_decoder_outputs(mel_specgrams, gate_outputs, alignments)

        return mel_specgrams, mel_specgram_lengths, gate_outputs, alignments


class Tacotron2(nn.Module):
    r"""Tacotron2 model based on the implementation from
    `Nvidia <https://github.com/NVIDIA/DeepLearningExamples/>`_.

    The original implementation was introduced in
    *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*
    [:footcite:`shen2018natural`].

    Args:
        mask_padding (bool, optional): Use mask padding (Default: ``False``).
        n_mels (int, optional): Number of mel bins (Default: ``80``).
        n_symbol (int, optional): Number of symbols for the input text (Default: ``148``).
        n_frames_per_step (int, optional): Number of frames processed per step, only 1 is supported (Default: ``1``).
        symbol_embedding_dim (int, optional): Input embedding dimension (Default: ``512``).
        encoder_n_convolution (int, optional): Number of encoder convolutions (Default: ``3``).
        encoder_kernel_size (int, optional): Encoder kernel size (Default: ``5``).
        encoder_embedding_dim (int, optional): Encoder embedding dimension (Default: ``512``).
        decoder_rnn_dim (int, optional): Number of units in decoder LSTM (Default: ``1024``).
        decoder_max_step (int, optional): Maximum number of output mel spectrograms (Default: ``2000``).
        decoder_dropout (float, optional): Dropout probability for decoder LSTM (Default: ``0.1``).
        decoder_early_stopping (bool, optional): Continue decoding after all samples are finished (Default: ``True``).
        attention_rnn_dim (int, optional): Number of units in attention LSTM (Default: ``1024``).
        attention_hidden_dim (int, optional): Dimension of attention hidden representation (Default: ``128``).
        attention_location_n_filter (int, optional): Number of filters for attention model (Default: ``32``).
        attention_location_kernel_size (int, optional): Kernel size for attention model (Default: ``31``).
        attention_dropout (float, optional): Dropout probability for attention LSTM (Default: ``0.1``).
        prenet_dim (int, optional): Number of ReLU units in prenet layers (Default: ``256``).
        postnet_n_convolution (int, optional): Number of postnet convolutions (Default: ``5``).
        postnet_kernel_size (int, optional): Postnet kernel size (Default: ``5``).
        postnet_embedding_dim (int, optional): Postnet embedding dimension (Default: ``512``).
        gate_threshold (float, optional): Probability threshold for stop token (Default: ``0.5``).
    """

    def __init__(
        self,
        mask_padding: bool = False,
        n_mels: int = 80,
        n_symbol: int = 148,
        n_frames_per_step: int = 1,
        symbol_embedding_dim: int = 512,
        encoder_embedding_dim: int = 512,
        encoder_n_convolution: int = 3,
        encoder_kernel_size: int = 5,
        decoder_rnn_dim: int = 1024,
        decoder_max_step: int = 2000,
        decoder_dropout: float = 0.1,
        decoder_early_stopping: bool = True,
        attention_rnn_dim: int = 1024,
        attention_hidden_dim: int = 128,
        attention_location_n_filter: int = 32,
        attention_location_kernel_size: int = 31,
        attention_dropout: float = 0.1,
        prenet_dim: int = 256,
        postnet_n_convolution: int = 5,
        postnet_kernel_size: int = 5,
        postnet_embedding_dim: int = 512,
        gate_threshold: float = 0.5,
    ) -> None:
        super().__init__()

        self.mask_padding = mask_padding
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.embedding = nn.Embedding(n_symbol, symbol_embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.encoder = _Encoder(encoder_embedding_dim, encoder_n_convolution, encoder_kernel_size)
        self.decoder = _Decoder(
            n_mels,
            n_frames_per_step,
            encoder_embedding_dim,
            decoder_rnn_dim,
            decoder_max_step,
            decoder_dropout,
            decoder_early_stopping,
            attention_rnn_dim,
            attention_hidden_dim,
            attention_location_n_filter,
            attention_location_kernel_size,
            attention_dropout,
            prenet_dim,
            gate_threshold,
        )
        self.postnet = _Postnet(n_mels, postnet_embedding_dim, postnet_kernel_size, postnet_n_convolution)

    def forward(
        self,
        tokens: Tensor,
        token_lengths: Tensor,
        mel_specgram: Tensor,
        mel_specgram_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Pass the input through the Tacotron2 model. This is in teacher
        forcing mode, which is generally used for training.

        The input ``tokens`` should be padded with zeros to length max of ``token_lengths``.
        The input ``mel_specgram`` should be padded with zeros to length max of ``mel_specgram_lengths``.

        Args:
            tokens (Tensor): The input tokens to Tacotron2 with shape `(n_batch, max of token_lengths)`.
            token_lengths (Tensor): The valid length of each sample in ``tokens`` with shape `(n_batch, )`.
            mel_specgram (Tensor): The target mel spectrogram
                with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
            mel_specgram_lengths (Tensor): The length of each mel spectrogram with shape `(n_batch, )`.

        Returns:
            [Tensor, Tensor, Tensor, Tensor]:
                Tensor
                    Mel spectrogram before Postnet with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    Mel spectrogram after Postnet with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    The output for stop token at each time step with shape `(n_batch, max of mel_specgram_lengths)`.
                Tensor
                    Sequence of attention weights from the decoder with
                    shape `(n_batch, max of mel_specgram_lengths, max of token_lengths)`.
        """

        embedded_inputs = self.embedding(tokens).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, token_lengths)
        mel_specgram, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_specgram, memory_lengths=token_lengths
        )

        mel_specgram_postnet = self.postnet(mel_specgram)
        mel_specgram_postnet = mel_specgram + mel_specgram_postnet

        if self.mask_padding:
            mask = _get_mask_from_lengths(mel_specgram_lengths)
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_specgram.masked_fill_(mask, 0.0)
            mel_specgram_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)

        return mel_specgram, mel_specgram_postnet, gate_outputs, alignments

    @torch.jit.export
    def infer(self, tokens: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Using Tacotron2 for inference. The input is a batch of encoded
        sentences (``tokens``) and its corresponding lengths (``lengths``). The
        output is the generated mel spectrograms, its corresponding lengths, and
        the attention weights from the decoder.

        The input `tokens` should be padded with zeros to length max of ``lengths``.

        Args:
            tokens (Tensor): The input tokens to Tacotron2 with shape `(n_batch, max of lengths)`.
            lengths (Tensor or None, optional):
                The valid length of each sample in ``tokens`` with shape `(n_batch, )`.
                If ``None``, it is assumed that the all the tokens are valid. Default: ``None``

        Returns:
            (Tensor, Tensor, Tensor):
                Tensor
                    The predicted mel spectrogram with shape `(n_batch, n_mels, max of mel_specgram_lengths)`.
                Tensor
                    The length of the predicted mel spectrogram with shape `(n_batch, )`.
                Tensor
                    Sequence of attention weights from the decoder with shape
                    `(n_batch, max of mel_specgram_lengths, max of lengths)`.
        """
        n_batch, max_length = tokens.shape
        if lengths is None:
            lengths = torch.tensor([max_length]).expand(n_batch).to(tokens.device, tokens.dtype)

        assert lengths is not None  # For TorchScript compiler

        embedded_inputs = self.embedding(tokens).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, lengths)
        mel_specgram, mel_specgram_lengths, _, alignments = self.decoder.infer(encoder_outputs, lengths)

        mel_outputs_postnet = self.postnet(mel_specgram)
        mel_outputs_postnet = mel_specgram + mel_outputs_postnet

        alignments = alignments.unfold(1, n_batch, n_batch).transpose(0, 2)

        return mel_outputs_postnet, mel_specgram_lengths, alignments
