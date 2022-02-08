from typing import List, Optional, Tuple

import torch
from torchaudio.models import Emformer


__all__ = ["RNNT", "emformer_rnnt_base", "emformer_rnnt_model"]


class _TimeReduction(torch.nn.Module):
    r"""Coalesces frames along time dimension into a
    fewer number of frames with higher feature dimensionality.

    Args:
        stride (int): number of frames to merge for each output frame.
    """

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass.

        B: batch size;
        T: maximum input sequence length in batch;
        D: feature dimension of each input sequence frame.

        Args:
            input (torch.Tensor): input sequences, with shape `(B, T, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output sequences, with shape
                    `(B, T  // stride, D * stride)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output sequences.
        """
        B, T, D = input.shape
        num_frames = T - (T % self.stride)
        input = input[:, :num_frames, :]
        lengths = lengths.div(self.stride, rounding_mode="trunc")
        T_max = num_frames // self.stride

        output = input.reshape(B, T_max, D * self.stride)
        output = output.contiguous()
        return output, lengths


class _CustomLSTM(torch.nn.Module):
    r"""Custom long-short-term memory (LSTM) block that applies layer normalization
    to internal nodes.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        layer_norm (bool, optional): if ``True``, enables layer normalization. (Default: ``False``)
        layer_norm_epsilon (float, optional):  value of epsilon to use in
            layer normalization layers (Default: 1e-5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_norm: bool = False,
        layer_norm_epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.x2g = torch.nn.Linear(input_dim, 4 * hidden_dim, bias=(not layer_norm))
        self.p2g = torch.nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        if layer_norm:
            self.c_norm = torch.nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon)
            self.g_norm = torch.nn.LayerNorm(4 * hidden_dim, eps=layer_norm_epsilon)
        else:
            self.c_norm = torch.nn.Identity()
            self.g_norm = torch.nn.Identity()

        self.hidden_dim = hidden_dim

    def forward(
        self, input: torch.Tensor, state: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        r"""Forward pass.

        B: batch size;
        T: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): with shape `(T, B, D)`.
            state (List[torch.Tensor] or None): list of tensors
                representing internal state generated in preceding invocation
                of ``forward``.

        Returns:
            (torch.Tensor, List[torch.Tensor]):
                torch.Tensor
                    output, with shape `(T, B, hidden_dim)`.
                List[torch.Tensor]
                    list of tensors representing internal state generated
                    in current invocation of ``forward``.
        """
        if state is None:
            B = input.size(1)
            h = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
            c = torch.zeros(B, self.hidden_dim, device=input.device, dtype=input.dtype)
        else:
            h, c = state

        gated_input = self.x2g(input)
        outputs = []
        for gates in gated_input.unbind(0):
            gates = gates + self.p2g(h)
            gates = self.g_norm(gates)
            input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
            input_gate = input_gate.sigmoid()
            forget_gate = forget_gate.sigmoid()
            cell_gate = cell_gate.tanh()
            output_gate = output_gate.sigmoid()
            c = forget_gate * c + input_gate * cell_gate
            c = self.c_norm(c)
            h = output_gate * c.tanh()
            outputs.append(h)

        output = torch.stack(outputs, dim=0)
        state = [h, c]

        return output, state


class _Transcriber(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) transcription network.

    Args:
        input_dim (int): feature dimension of each input sequence element.
        output_dim (int): feature dimension of each output sequence element.
        segment_length (int): length of input segment expressed as number of frames.
        right_context_length (int): length of right context expressed as number of frames.
        time_reduction_input_dim (int): dimension to scale each element in input sequences to
            prior to applying time reduction block.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        transformer_num_heads (int): number of attention heads in each Emformer layer.
        transformer_ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        transformer_num_layers (int): number of Emformer layers to instantiate.
        transformer_left_context_length (int): length of left context.
        transformer_dropout (float, optional): transformer dropout probability. (Default: 0.0)
        transformer_activation (str, optional): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        transformer_max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        transformer_weight_init_scale_strategy (str, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        transformer_tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        segment_length: int,
        right_context_length: int,
        time_reduction_input_dim: int,
        time_reduction_stride: int,
        transformer_num_heads: int,
        transformer_ffn_dim: int,
        transformer_num_layers: int,
        transformer_left_context_length: int,
        transformer_dropout: float = 0.0,
        transformer_activation: str = "relu",
        transformer_max_memory_size: int = 0,
        transformer_weight_init_scale_strategy: str = "depthwise",
        transformer_tanh_on_mem: bool = False,
    ) -> None:
        super().__init__()
        self.input_linear = torch.nn.Linear(
            input_dim,
            time_reduction_input_dim,
            bias=False,
        )
        self.time_reduction = _TimeReduction(time_reduction_stride)
        transformer_input_dim = time_reduction_input_dim * time_reduction_stride
        self.transformer = Emformer(
            transformer_input_dim,
            transformer_num_heads,
            transformer_ffn_dim,
            transformer_num_layers,
            segment_length // time_reduction_stride,
            dropout=transformer_dropout,
            activation=transformer_activation,
            left_context_length=transformer_left_context_length,
            right_context_length=right_context_length // time_reduction_stride,
            max_memory_size=transformer_max_memory_size,
            weight_init_scale_strategy=transformer_weight_init_scale_strategy,
            tanh_on_mem=transformer_tanh_on_mem,
        )
        self.output_linear = torch.nn.Linear(transformer_input_dim, output_dim)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum input sequence length in batch;
        D: feature dimension of each input sequence frame (input_dim).

        Args:
            input (torch.Tensor): input frame sequences right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output frame sequences.
        """
        input_linear_out = self.input_linear(input)
        time_reduction_out, time_reduction_lengths = self.time_reduction(input_linear_out, lengths)
        transformer_out, transformer_lengths = self.transformer(time_reduction_out, time_reduction_lengths)
        output_linear_out = self.output_linear(transformer_out)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, transformer_lengths

    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for inference.

        B: batch size;
        T: maximum input sequence segment length in batch;
        D: feature dimension of each input sequence frame (input_dim).

        Args:
            input (torch.Tensor): input frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``infer``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output input lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation
                    of ``infer``.
        """
        input_linear_out = self.input_linear(input)
        time_reduction_out, time_reduction_lengths = self.time_reduction(input_linear_out, lengths)
        (
            transformer_out,
            transformer_lengths,
            transformer_states,
        ) = self.transformer.infer(time_reduction_out, time_reduction_lengths, states)
        output_linear_out = self.output_linear(transformer_out)
        layer_norm_out = self.layer_norm(output_linear_out)
        return layer_norm_out, transformer_lengths, transformer_states


class _Predictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Args:
        num_symbols (int): size of target token lexicon.
        output_dim (int): feature dimension of each output sequence element.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_layer_norm (bool, optional): if ``True``, enables layer normalization
            for LSTM layers. (Default: ``False``)
        lstm_layer_norm_epsilon (float, optional): value of epsilon to use in
            LSTM layer normalization layers. (Default: 1e-5)
        lstm_dropout (float, optional): LSTM dropout probability. (Default: 0.0)

    """

    def __init__(
        self,
        num_symbols: int,
        output_dim: int,
        symbol_embedding_dim: int,
        num_lstm_layers: int,
        lstm_layer_norm: bool = False,
        lstm_layer_norm_epsilon: float = 1e-5,
        lstm_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_symbols, symbol_embedding_dim)
        self.input_layer_norm = torch.nn.LayerNorm(symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                _CustomLSTM(
                    symbol_embedding_dim,
                    symbol_embedding_dim,
                    layer_norm=lstm_layer_norm,
                    layer_norm_epsilon=lstm_layer_norm_epsilon,
                )
                for idx in range(num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=lstm_dropout)
        self.linear = torch.nn.Linear(symbol_embedding_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.lstm_dropout = lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else state[layer_idx])
            lstm_out = self.dropout(lstm_out)
            state_out.append(lstm_state_out)

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), lengths, state_out


class _Joiner(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        relu_out = self.relu(joint_encodings)
        output = self.linear(relu_out)
        return output, source_lengths, target_lengths


class RNNT(torch.nn.Module):
    r"""torchaudio.models.RNNT()

    Recurrent neural network transducer (RNN-T) model.

    Note:
        To build the model, please use one of the factory functions.

    Args:
        transcriber (torch.nn.Module): transcription network.
        predictor (torch.nn.Module): prediction network.
        joiner (torch.nn.Module): joint network.
    """

    def __init__(self, transcriber: _Transcriber, predictor: _Predictor, joiner: _Joiner) -> None:
        super().__init__()
        self.transcriber = transcriber
        self.predictor = predictor
        self.joiner = joiner

    def forward(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictor_state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: feature dimension of each source sequence element.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            predictor_state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing prediction network internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    joint network output, with shape
                    `(B, max output source length, max output target length, output_dim (number of target symbols))`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing prediction network internal state generated in current invocation
                    of ``forward``.
        """
        source_encodings, source_lengths = self.transcriber(
            input=sources,
            lengths=source_lengths,
        )
        target_encodings, target_lengths, predictor_state = self.predictor(
            input=targets,
            lengths=target_lengths,
            state=predictor_state,
        )
        output, source_lengths, target_lengths = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )

        return (
            output,
            source_lengths,
            target_lengths,
            predictor_state,
        )

    @torch.jit.export
    def transcribe_streaming(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Applies transcription network to sources in streaming mode.

        B: batch size;
        T: maximum source sequence segment length in batch;
        D: feature dimension of each source sequence frame.

        Args:
            sources (torch.Tensor): source frame sequence segments right-padded with right context, with
                shape `(B, T + right context length, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing transcription network internal state generated in preceding invocation
                of ``transcribe_streaming``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing transcription network internal state generated in current invocation
                    of ``transcribe_streaming``.
        """
        return self.transcriber.infer(sources, source_lengths, state)

    @torch.jit.export
    def transcribe(
        self,
        sources: torch.Tensor,
        source_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Applies transcription network to sources in non-streaming mode.

        B: batch size;
        T: maximum source sequence length in batch;
        D: feature dimension of each source sequence frame.

        Args:
            sources (torch.Tensor): source frame sequences right-padded with right context, with
                shape `(B, T + right context length, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``sources``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frame sequences, with
                    shape `(B, T // time_reduction_stride, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output frame sequences.
        """
        return self.transcriber(sources, source_lengths)

    @torch.jit.export
    def predict(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Applies prediction network to targets.

        B: batch size;
        U: maximum target sequence length in batch;
        D: feature dimension of each target sequence frame.

        Args:
            targets (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``targets``.
            state (List[List[torch.Tensor]] or None): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``predict``.

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output frame sequences, with shape `(B, U, output_dim)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``predict``.
        """
        return self.predictor(input=targets, lengths=target_lengths, state=state)

    @torch.jit.export
    def join(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Applies joint network to source and target encodings.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        output, source_lengths, target_lengths = self.joiner(
            source_encodings=source_encodings,
            source_lengths=source_lengths,
            target_encodings=target_encodings,
            target_lengths=target_lengths,
        )
        return output, source_lengths, target_lengths


def emformer_rnnt_model(
    *,
    input_dim: int,
    encoding_dim: int,
    num_symbols: int,
    segment_length: int,
    right_context_length: int,
    time_reduction_input_dim: int,
    time_reduction_stride: int,
    transformer_num_heads: int,
    transformer_ffn_dim: int,
    transformer_num_layers: int,
    transformer_dropout: float,
    transformer_activation: str,
    transformer_left_context_length: int,
    transformer_max_memory_size: int,
    transformer_weight_init_scale_strategy: str,
    transformer_tanh_on_mem: bool,
    symbol_embedding_dim: int,
    num_lstm_layers: int,
    lstm_layer_norm: bool,
    lstm_layer_norm_epsilon: float,
    lstm_dropout: float,
) -> RNNT:
    r"""Builds Emformer-based recurrent neural network transducer (RNN-T) model.

    Note:
        For non-streaming inference, the expectation is for `transcribe` to be called on input
        sequences right-concatenated with `right_context_length` frames.

        For streaming inference, the expectation is for `transcribe_streaming` to be called
        on input chunks comprising `segment_length` frames right-concatenated with `right_context_length`
        frames.

    Args:
        input_dim (int): dimension of input sequence frames passed to transcription network.
        encoding_dim (int): dimension of transcription- and prediction-network-generated encodings
            passed to joint network.
        num_symbols (int): cardinality of set of target tokens.
        segment_length (int): length of input segment expressed as number of frames.
        right_context_length (int): length of right context expressed as number of frames.
        time_reduction_input_dim (int): dimension to scale each element in input sequences to
            prior to applying time reduction block.
        time_reduction_stride (int): factor by which to reduce length of input sequence.
        transformer_num_heads (int): number of attention heads in each Emformer layer.
        transformer_ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        transformer_num_layers (int): number of Emformer layers to instantiate.
        transformer_left_context_length (int): length of left context considered by Emformer.
        transformer_dropout (float): Emformer dropout probability.
        transformer_activation (str): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu").
        transformer_max_memory_size (int): maximum number of memory elements to use.
        transformer_weight_init_scale_strategy (str): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``).
        transformer_tanh_on_mem (bool): if ``True``, applies tanh to memory elements.
        symbol_embedding_dim (int): dimension of each target token embedding.
        num_lstm_layers (int): number of LSTM layers to instantiate.
        lstm_layer_norm (bool): if ``True``, enables layer normalization for LSTM layers.
        lstm_layer_norm_epsilon (float): value of epsilon to use in LSTM layer normalization layers.
        lstm_dropout (float): LSTM dropout probability.

    Returns:
        RNNT:
            Emformer RNN-T model.
    """
    transcriber = _Transcriber(
        input_dim=input_dim,
        output_dim=encoding_dim,
        segment_length=segment_length,
        right_context_length=right_context_length,
        time_reduction_input_dim=time_reduction_input_dim,
        time_reduction_stride=time_reduction_stride,
        transformer_num_heads=transformer_num_heads,
        transformer_ffn_dim=transformer_ffn_dim,
        transformer_num_layers=transformer_num_layers,
        transformer_dropout=transformer_dropout,
        transformer_activation=transformer_activation,
        transformer_left_context_length=transformer_left_context_length,
        transformer_max_memory_size=transformer_max_memory_size,
        transformer_weight_init_scale_strategy=transformer_weight_init_scale_strategy,
        transformer_tanh_on_mem=transformer_tanh_on_mem,
    )
    predictor = _Predictor(
        num_symbols,
        encoding_dim,
        symbol_embedding_dim=symbol_embedding_dim,
        num_lstm_layers=num_lstm_layers,
        lstm_layer_norm=lstm_layer_norm,
        lstm_layer_norm_epsilon=lstm_layer_norm_epsilon,
        lstm_dropout=lstm_dropout,
    )
    joiner = _Joiner(encoding_dim, num_symbols)
    return RNNT(transcriber, predictor, joiner)


def emformer_rnnt_base(num_symbols: int) -> RNNT:
    r"""Builds basic version of Emformer RNN-T model.

    Args:
        num_symbols (int): The size of target token lexicon.

    Returns:
        RNNT:
            Emformer RNN-T model.
    """
    return emformer_rnnt_model(
        input_dim=80,
        encoding_dim=1024,
        num_symbols=num_symbols,
        segment_length=16,
        right_context_length=4,
        time_reduction_input_dim=128,
        time_reduction_stride=4,
        transformer_num_heads=8,
        transformer_ffn_dim=2048,
        transformer_num_layers=20,
        transformer_dropout=0.1,
        transformer_activation="gelu",
        transformer_left_context_length=30,
        transformer_max_memory_size=0,
        transformer_weight_init_scale_strategy="depthwise",
        transformer_tanh_on_mem=True,
        symbol_embedding_dim=512,
        num_lstm_layers=3,
        lstm_layer_norm=True,
        lstm_layer_norm_epsilon=1e-3,
        lstm_dropout=0.3,
    )
