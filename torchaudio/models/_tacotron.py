from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "_ConvNorm",
    "_LinearNorm",
    "_LocationLayer",
    "_Prenet",
    "_Attention",
    "_Encoder",
    "_Decoder",
]


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


class _LinearNorm(nn.Module):
    r"""Linear layer

    Args:
        n_input: the number of input channels.
        n_output: the number of output channels.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super(_LinearNorm, self).__init__()

        self.linear_layer = torch.nn.Linear(n_input, n_output, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through the _LinearNorm layer.
        """

        return self.linear_layer(x)


class _Prenet(nn.Module):
    r"""Prenet module
    """

    def __init__(self, n_input: Tensor, sizes: Tensor) -> None:
        super(_Prenet, self).__init__()

        in_sizes = [n_input] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                _LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Pass the input through the _Prenet layer.

        Args:
            x: the input sequence to the _Prenet layer
        """

        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class _LocationLayer(nn.Module):
    r"""Location layer for attention processing

    Args:
        attention_n_filters: the number of filters for location-sensitive attention
        attention_kernel_size: the kernel size for location-sensitive attention
        n_attention: the number of dimension of attention hidden representation
    """

    def __init__(
        self, attention_n_filters: int, attention_kernel_size: int, n_attention: int
    ) -> None:
        super(_LocationLayer, self).__init__()

        padding = int((attention_kernel_size - 1) / 2)

        self.location_conv = _ConvNorm(
            n_input=2,
            n_output=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )

        self.location_dense = _LinearNorm(
            n_input=attention_n_filters,
            n_output=n_attention,
            bias=False,
            w_init_gain="tanh",
        )

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        r"""Pass the input through the _LocationLayer.
        """

        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class _Attention(nn.Module):
    """ Attention layer

    Args:
        n_attention_rnn: the number of filters for location-sensitive attention
        n_embedding: the dimension of embedding representation
        n_attention: the dimension of attention hidden representation
        attention_location_n_filters: the number of filters for location-sensitive attention
        attention_location_kernel_size: the kernel size for location-sensitive attention
    """

    def __init__(
        self,
        n_attention_rnn: int,
        n_embedding: int,
        n_attention: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
    ) -> None:
        super(_Attention, self).__init__()

        self.query_layer = _LinearNorm(
            n_attention_rnn, n_attention, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = _LinearNorm(
            n_embedding, n_attention, bias=False, w_init_gain="tanh"
        )
        self.v = _LinearNorm(n_attention, 1, bias=False)
        self.location_layer = _LocationLayer(
            attention_location_n_filters, attention_location_kernel_size, n_attention
        )
        self.score_mask_value = -float("inf")

    def _get_alignment_energies(
        self, query: Tensor, processed_memory: Tensor, attention_weights_cat: Tensor
    ) -> Tensor:
        r"""

        Args:
            query: decoder output (n_batch, n_mel_channels * n_frames_per_step)
            processed_memory: processed encoder outputs (B, T_in, n_attention)
            attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        Return:
            alignment (n_batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )

        energies = energies.squeeze(2)
        return energies

    def forward(
        self,
        attention_hidden_state: Tensor,
        memory: Tensor,
        processed_memory: Tensor,
        attention_weights_cat: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor]:
        r"""Pass the input through the _Attention layer.

        Args:
            attention_hidden_state: attention rnn last output
            memory: encoder outputs
            processed_memory: processed encoder outputs
            attention_weights_cat: previous and cummulative attention weights
            mask: binary mask for padded data
        """

        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        alignment = alignment.masked_fill(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


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


class _Decoder(nn.Module):
    r"""Decoder module

        Args:
            n_mel_channels: the number of bins in mel-spectrograms
            n_frames_per_step: the number of frames per step
            n_encoder_embedding: the number of embedding dimensions in the encoder
            n_attention: the number of dimension of attention hidden representation
            attention_location_n_filters: the number of filters for location-sensitive attention
            attention_location_kernel_size: the kernel size for location-sensitive attention
            n_attention_rnn: the number of units in attention LSTM
            n_decoder_rnn: the number of units in decoder LSTM
            n_prenet: the number of ReLU units in prenet layers
            max_decoder_steps: the maximum number of output mel spectrograms
            gate_threshold: the probability threshold for stop token
            p_attention_dropout: the dropout probability for attention LSTM
            p_decoder_dropout: the dropout probability for decoder LSTM
            early_stopping: if early stop during decoding
    """

    def __init__(
        self,
        n_mel_channels: int,
        n_frames_per_step: int,
        n_encoder_embedding: int,
        n_attention: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
        n_attention_rnn: int,
        n_decoder_rnn: int,
        n_prenet: int,
        max_decoder_steps: int,
        gate_threshold: float,
        p_attention_dropout: float,
        p_decoder_dropout: float,
        early_stopping: bool,
    ) -> None:
        super(_Decoder, self).__init__()

        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.n_encoder_embedding = n_encoder_embedding
        self.n_attention_rnn = n_attention_rnn
        self.n_decoder_rnn = n_decoder_rnn
        self.n_prenet = n_prenet
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        self.prenet = _Prenet(n_mel_channels * n_frames_per_step, [n_prenet, n_prenet])

        self.attention_rnn = nn.LSTMCell(
            n_prenet + n_encoder_embedding, n_attention_rnn
        )

        self.attention_layer = _Attention(
            n_attention_rnn,
            n_encoder_embedding,
            n_attention,
            attention_location_n_filters,
            attention_location_kernel_size,
        )

        self.decoder_rnn = nn.LSTMCell(
            n_attention_rnn + n_encoder_embedding, n_decoder_rnn, 1
        )

        self.linear_projection = _LinearNorm(
            n_decoder_rnn + n_encoder_embedding, n_mel_channels * n_frames_per_step
        )

        self.gate_layer = _LinearNorm(
            n_decoder_rnn + n_encoder_embedding, 1, bias=True, w_init_gain="sigmoid"
        )

    def _get_mask_from_lengths(self, lengths: Tensor) -> Tensor:
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
        mask = (ids < lengths.unsqueeze(1)).byte()
        mask = torch.le(mask, 0)
        return mask

    def _get_go_frame(self, memory: Tensor) -> Tensor:
        r""" Get all zeros frames to use as first decoder input

        Args:
            memory: decoder outputs

        Return:
            decoder_input: all zeros frames
        """

        B = memory.size(0)
        dtype = memory.dtype
        device = memory.device
        decoder_input = torch.zeros(
            B, self.n_mel_channels * self.n_frames_per_step, dtype=dtype, device=device
        )
        return decoder_input

    def _initialize_decoder_states(self, memory: Tensor) -> Tensor:

        r"""Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory

        Args:
            memory: encoder outputs.
            mask: mask for padded data if training, expects None for inference
        """

        B = memory.size(0)
        MAX_TIME = memory.size(1)
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(
            B, self.n_attention_rnn, dtype=dtype, device=device
        )
        attention_cell = torch.zeros(
            B, self.n_attention_rnn, dtype=dtype, device=device
        )

        decoder_hidden = torch.zeros(B, self.n_decoder_rnn, dtype=dtype, device=device)
        decoder_cell = torch.zeros(B, self.n_decoder_rnn, dtype=dtype, device=device)

        attention_weights = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_context = torch.zeros(
            B, self.n_encoder_embedding, dtype=dtype, device=device
        )

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
        r""" Prepares decoder inputs

        Args:
            decoder_inputs: inputs used for teacher-forced training

        Return:
            decoder_inputs: processed decoder inputs
        """

        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def _parse_decoder_outputs(
        self, mel_outputs: Tensor, gate_outputs: Tensor, alignments: Tensor
    ) -> Tuple[Tensor]:
        r""" Prepares decoder outputs for output

        Args:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            attention_weights: sequence of attention weights from the decoder

        Return:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            attention_weights: sequence of attention weights from the decoder
        """

        # (T_out, B) -> (B, T_out)
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # decouple frames per step
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)
        mel_outputs = mel_outputs.view(*shape)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def _decode(
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
    ) -> Tuple[Tensor]:
        r""" Decoder step using stored states, attention and memory

        Args:
            decoder_input: previous mel output

        Return:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            attention_weights: sequence of attention weights from the decoder
        """

        cell_input = torch.cat((decoder_input, attention_context), -1)

        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, (attention_hidden, attention_cell)
        )
        attention_hidden = F.dropout(
            attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1
        )
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory, attention_weights_cat, mask
        )

        attention_weights_cum += attention_weights
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, (decoder_hidden, decoder_cell)
        )
        decoder_hidden = F.dropout(
            decoder_hidden, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (decoder_hidden, attention_context), dim=1
        )
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
        self, memory: Tensor, decoder_inputs: Tensor, memory_lengths: Tensor
    ) -> Tuple[Tensor]:
        r""" Decoder forward pass for training

        Args:
            memory: Encoder outputs
            decoder_inputs: Decoder inputs for teacher forcing
            memory_lengths: Encoder output lengths for attention masking

        Return:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            attention_weights: sequence of attention weights from the decoder
        """

        decoder_input = self._get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self._parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self._prenet(decoder_inputs)

        mask = self._get_mask_from_lengths(memory_lengths)

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
            ) = self._decode(
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
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self._parse_decoder_outputs(
            torch.stack(mel_outputs), torch.stack(gate_outputs), torch.stack(alignments)
        )

        return mel_outputs, gate_outputs, alignments

    def infer(self, memory: Tensor, memory_lengths: Tensor) -> Tuple[Tensor]:
        r""" Decoder forward pass for inference

        Args:
            memory: Encoder outputs

        Return:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            attention_weights: sequence of attention weights from the decoder
        """

        decoder_input = self._get_go_frame(memory)

        mask = self._get_mask_from_lengths(memory_lengths)

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

        mel_lengths = torch.zeros(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )
        not_finished = torch.ones(
            [memory.size(0)], dtype=torch.int32, device=memory.device
        )

        mel_outputs, gate_outputs, alignments = (
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )
        first_iter = True
        while True:
            decoder_input = self._prenet(decoder_input)

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
            ) = self._decode(
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

            if first_iter:
                mel_outputs = mel_output.unsqueeze(0)
                gate_outputs = gate_output
                alignments = attention_weights
                first_iter = False
            else:
                mel_outputs = torch.cat((mel_outputs, mel_output.unsqueeze(0)), dim=0)
                gate_outputs = torch.cat((gate_outputs, gate_output), dim=0)
                alignments = torch.cat((alignments, attention_weights), dim=0)

            dec = (
                torch.le(torch.sigmoid(gate_output), self.gate_threshold)
                .to(torch.int32)
                .squeeze(1)
            )

            not_finished = not_finished * dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0:
                break
            if len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self._parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments, mel_lengths
