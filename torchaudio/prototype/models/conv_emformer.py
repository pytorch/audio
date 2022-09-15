import math
from typing import List, Optional, Tuple

import torch
from torchaudio.models.emformer import _EmformerAttention, _EmformerImpl, _get_weight_init_gains


def _get_activation_module(activation: str) -> torch.nn.Module:
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    elif activation == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation {activation}")


class _ResidualContainer(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, output_weight: int):
        super().__init__()
        self.module = module
        self.output_weight = output_weight

    def forward(self, input: torch.Tensor):
        output = self.module(input)
        return output * self.output_weight + input


class _ConvolutionModule(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        segment_length: int,
        right_context_length: int,
        kernel_size: int,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.segment_length = segment_length
        self.right_context_length = right_context_length
        self.state_size = kernel_size - 1

        self.pre_conv = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, 2 * input_dim, bias=True), torch.nn.GLU()
        )
        self.conv = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=input_dim,
        )
        self.post_conv = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            _get_activation_module(activation),
            torch.nn.Linear(input_dim, input_dim, bias=True),
            torch.nn.Dropout(p=dropout),
        )

    def _split_right_context(self, utterance: torch.Tensor, right_context: torch.Tensor) -> torch.Tensor:
        T, B, D = right_context.size()
        if T % self.right_context_length != 0:
            raise ValueError("Tensor length should be divisible by its right context length")
        num_segments = T // self.right_context_length
        # (num_segments, right context length, B, D)
        right_context_segments = right_context.reshape(num_segments, self.right_context_length, B, D)
        right_context_segments = right_context_segments.permute(0, 2, 1, 3).reshape(
            num_segments * B, self.right_context_length, D
        )

        pad_segments = []  # [(kernel_size - 1, B, D), ...]
        for seg_idx in range(num_segments):
            end_idx = min(self.state_size + (seg_idx + 1) * self.segment_length, utterance.size(0))
            start_idx = end_idx - self.state_size
            pad_segments.append(utterance[start_idx:end_idx, :, :])

        pad_segments = torch.cat(pad_segments, dim=1).permute(1, 0, 2)  # (num_segments * B, kernel_size - 1, D)
        return torch.cat([pad_segments, right_context_segments], dim=1).permute(0, 2, 1)

    def _merge_right_context(self, right_context: torch.Tensor, B: int) -> torch.Tensor:
        # (num_segments * B, D, right_context_length)
        right_context = right_context.reshape(-1, B, self.input_dim, self.right_context_length)
        right_context = right_context.permute(0, 3, 1, 2)
        return right_context.reshape(-1, B, self.input_dim)  # (right_context_length * num_segments, B, D)

    def forward(
        self, utterance: torch.Tensor, right_context: torch.Tensor, state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = torch.cat((right_context, utterance))  # input: (T, B, D)
        x = self.pre_conv(input)
        x_right_context, x_utterance = x[: right_context.size(0), :, :], x[right_context.size(0) :, :, :]
        x_utterance = x_utterance.permute(1, 2, 0)  # (B, D, T_utterance)

        if state is None:
            state = torch.zeros(
                input.size(1),
                input.size(2),
                self.state_size,
                device=input.device,
                dtype=input.dtype,
            )  # (B, D, T)
        state_x_utterance = torch.cat([state, x_utterance], dim=2)

        conv_utterance = self.conv(state_x_utterance)  # (B, D, T_utterance)
        conv_utterance = conv_utterance.permute(2, 0, 1)

        if self.right_context_length > 0:
            # (B * num_segments, D, right_context_length + kernel_size - 1)
            right_context_block = self._split_right_context(state_x_utterance.permute(2, 0, 1), x_right_context)
            conv_right_context_block = self.conv(right_context_block)  # (B * num_segments, D, right_context_length)
            # (T_right_context, B, D)
            conv_right_context = self._merge_right_context(conv_right_context_block, input.size(1))
            y = torch.cat([conv_right_context, conv_utterance], dim=0)
        else:
            y = conv_utterance

        output = self.post_conv(y) + input
        new_state = state_x_utterance[:, :, -self.state_size :]
        return output[right_context.size(0) :], output[: right_context.size(0)], new_state

    def infer(
        self, utterance: torch.Tensor, right_context: torch.Tensor, state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = torch.cat((utterance, right_context))
        x = self.pre_conv(input)  # (T, B, D)
        x = x.permute(1, 2, 0)  # (B, D, T)

        if state is None:
            state = torch.zeros(
                input.size(1),
                input.size(2),
                self.state_size,
                device=input.device,
                dtype=input.dtype,
            )  # (B, D, T)
        state_x = torch.cat([state, x], dim=2)
        conv_out = self.conv(state_x)
        conv_out = conv_out.permute(2, 0, 1)  # T, B, D
        output = self.post_conv(conv_out) + input
        new_state = state_x[:, :, -self.state_size - right_context.size(0) : -right_context.size(0)]
        return output[: utterance.size(0)], output[utterance.size(0) :], new_state


class _ConvEmformerLayer(torch.nn.Module):
    r"""Convolution-augmented Emformer layer that constitutes ConvEmformer.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        ffn_dim: (int): hidden layer dimension of feedforward network.
        segment_length (int): length of each input segment.
        kernel_size (int): size of kernel to use in convolution module.
        dropout (float, optional): dropout probability. (Default: 0.0)
        ffn_activation (str, optional): activation function to use in feedforward network.
            Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        right_context_length (int, optional): length of right context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        weight_init_gain (float or None, optional): scale factor to apply when initializing
            attention module parameters. (Default: ``None``)
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
        conv_activation (str, optional): activation function to use in convolution module.
            Must be one of ("relu", "gelu", "silu"). (Default: "silu")
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        segment_length: int,
        kernel_size: int,
        dropout: float = 0.0,
        ffn_activation: str = "relu",
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        weight_init_gain: Optional[float] = None,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
        conv_activation: str = "silu",
    ):
        super().__init__()
        # TODO: implement talking heads attention.
        self.attention = _EmformerAttention(
            input_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            weight_init_gain=weight_init_gain,
            tanh_on_mem=tanh_on_mem,
            negative_inf=negative_inf,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.memory_op = torch.nn.AvgPool1d(kernel_size=segment_length, stride=segment_length, ceil_mode=True)

        activation_module = _get_activation_module(ffn_activation)
        self.ffn0 = _ResidualContainer(
            torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, ffn_dim),
                activation_module,
                torch.nn.Dropout(dropout),
                torch.nn.Linear(ffn_dim, input_dim),
                torch.nn.Dropout(dropout),
            ),
            0.5,
        )
        self.ffn1 = _ResidualContainer(
            torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, ffn_dim),
                activation_module,
                torch.nn.Dropout(dropout),
                torch.nn.Linear(ffn_dim, input_dim),
                torch.nn.Dropout(dropout),
            ),
            0.5,
        )
        self.layer_norm_input = torch.nn.LayerNorm(input_dim)
        self.layer_norm_output = torch.nn.LayerNorm(input_dim)

        self.conv = _ConvolutionModule(
            input_dim=input_dim,
            kernel_size=kernel_size,
            activation=conv_activation,
            dropout=dropout,
            segment_length=segment_length,
            right_context_length=right_context_length,
        )

        self.left_context_length = left_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.use_mem = max_memory_size > 0

    def _init_state(self, batch_size: int, device: Optional[torch.device]) -> List[torch.Tensor]:
        empty_memory = torch.zeros(self.max_memory_size, batch_size, self.input_dim, device=device)
        left_context_key = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        left_context_val = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        past_length = torch.zeros(1, batch_size, dtype=torch.int32, device=device)
        conv_cache = torch.zeros(
            batch_size,
            self.input_dim,
            self.kernel_size - 1,
            device=device,
        )
        return [empty_memory, left_context_key, left_context_val, past_length, conv_cache]

    def _unpack_state(self, state: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        past_length = state[3][0][0].item()
        past_left_context_length = min(self.left_context_length, past_length)
        past_mem_length = min(self.max_memory_size, math.ceil(past_length / self.segment_length))
        pre_mems = state[0][self.max_memory_size - past_mem_length :]
        lc_key = state[1][self.left_context_length - past_left_context_length :]
        lc_val = state[2][self.left_context_length - past_left_context_length :]
        conv_cache = state[4]
        return pre_mems, lc_key, lc_val, conv_cache

    def _pack_state(
        self,
        next_k: torch.Tensor,
        next_v: torch.Tensor,
        update_length: int,
        mems: torch.Tensor,
        conv_cache: torch.Tensor,
        state: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        new_k = torch.cat([state[1], next_k])
        new_v = torch.cat([state[2], next_v])
        state[0] = torch.cat([state[0], mems])[-self.max_memory_size :]
        state[1] = new_k[new_k.shape[0] - self.left_context_length :]
        state[2] = new_v[new_v.shape[0] - self.left_context_length :]
        state[3] = state[3] + update_length
        state[4] = conv_cache
        return state

    def _apply_pre_attention(
        self, utterance: torch.Tensor, right_context: torch.Tensor, summary: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([right_context, utterance, summary])
        ffn0_out = self.ffn0(x)
        layer_norm_input_out = self.layer_norm_input(ffn0_out)
        layer_norm_input_right_context, layer_norm_input_utterance, layer_norm_input_summary = (
            layer_norm_input_out[: right_context.size(0)],
            layer_norm_input_out[right_context.size(0) : right_context.size(0) + utterance.size(0)],
            layer_norm_input_out[right_context.size(0) + utterance.size(0) :],
        )
        return ffn0_out, layer_norm_input_right_context, layer_norm_input_utterance, layer_norm_input_summary

    def _apply_post_attention(
        self,
        rc_output: torch.Tensor,
        ffn0_out: torch.Tensor,
        conv_cache: Optional[torch.Tensor],
        rc_length: int,
        utterance_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.dropout(rc_output) + ffn0_out[: rc_length + utterance_length]
        conv_utterance, conv_right_context, conv_cache = self.conv(result[rc_length:], result[:rc_length], conv_cache)
        result = torch.cat([conv_right_context, conv_utterance])
        result = self.ffn1(result)
        result = self.layer_norm_output(result)
        output_utterance, output_right_context = result[rc_length:], result[:rc_length]
        return output_utterance, output_right_context, conv_cache

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            attention_mask (torch.Tensor): attention mask for underlying attention module.

        Returns:
            (Tensor, Tensor, Tensor):
                Tensor
                    encoded utterance frames, with shape `(T, B, D)`.
                Tensor
                    updated right context frames, with shape `(R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)

        (
            ffn0_out,
            layer_norm_input_right_context,
            layer_norm_input_utterance,
            layer_norm_input_summary,
        ) = self._apply_pre_attention(utterance, right_context, summary)

        rc_output, output_mems = self.attention(
            utterance=layer_norm_input_utterance,
            lengths=lengths,
            right_context=layer_norm_input_right_context,
            summary=layer_norm_input_summary,
            mems=mems,
            attention_mask=attention_mask,
        )

        output_utterance, output_right_context, _ = self._apply_post_attention(
            rc_output, ffn0_out, None, right_context.size(0), utterance.size(0)
        )

        return output_utterance, output_right_context, output_mems

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        state: Optional[List[torch.Tensor]],
        mems: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        r"""Forward pass for inference.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            state (List[torch.Tensor] or None): list of tensors representing layer internal state
                generated in preceding invocation of ``infer``.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.

        Returns:
            (Tensor, Tensor, List[torch.Tensor], Tensor):
                Tensor
                    encoded utterance frames, with shape `(T, B, D)`.
                Tensor
                    updated right context frames, with shape `(R, B, D)`.
                List[Tensor]
                    list of tensors representing layer internal state
                    generated in current invocation of ``infer``.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[:1]
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)

        (
            ffn0_out,
            layer_norm_input_right_context,
            layer_norm_input_utterance,
            layer_norm_input_summary,
        ) = self._apply_pre_attention(utterance, right_context, summary)

        if state is None:
            state = self._init_state(layer_norm_input_utterance.size(1), device=layer_norm_input_utterance.device)
        pre_mems, lc_key, lc_val, conv_cache = self._unpack_state(state)

        rc_output, next_m, next_k, next_v = self.attention.infer(
            utterance=layer_norm_input_utterance,
            lengths=lengths,
            right_context=layer_norm_input_right_context,
            summary=layer_norm_input_summary,
            mems=pre_mems,
            left_context_key=lc_key,
            left_context_val=lc_val,
        )

        output_utterance, output_right_context, conv_cache = self._apply_post_attention(
            rc_output, ffn0_out, conv_cache, right_context.size(0), utterance.size(0)
        )
        output_state = self._pack_state(next_k, next_v, utterance.size(0), mems, conv_cache, state)
        return output_utterance, output_right_context, output_state, next_m


class ConvEmformer(_EmformerImpl):
    r"""Implements the convolution-augmented streaming transformer architecture introduced in
    *Streaming Transformer Transducer based Speech Recognition Using Non-Causal Convolution*
    :cite:`9747706`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each ConvEmformer layer.
        ffn_dim (int): hidden layer dimension of each ConvEmformer layer's feedforward network.
        num_layers (int): number of ConvEmformer layers to instantiate.
        segment_length (int): length of each input segment.
        kernel_size (int): size of kernel to use in convolution modules.
        dropout (float, optional): dropout probability. (Default: 0.0)
        ffn_activation (str, optional): activation function to use in feedforward networks.
            Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        right_context_length (int, optional): length of right context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        weight_init_scale_strategy (str or None, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
        conv_activation (str, optional): activation function to use in convolution modules.
            Must be one of ("relu", "gelu", "silu"). (Default: "silu")

    Examples:
        >>> conv_emformer = ConvEmformer(80, 4, 1024, 12, 16, 8, right_context_length=4)
        >>> input = torch.rand(10, 200, 80)
        >>> lengths = torch.randint(1, 200, (10,))
        >>> output, lengths = conv_emformer(input, lengths)
        >>> input = torch.rand(4, 20, 80)
        >>> lengths = torch.ones(4) * 20
        >>> output, lengths, states = conv_emformer.infer(input, lengths, None)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        segment_length: int,
        kernel_size: int,
        dropout: float = 0.0,
        ffn_activation: str = "relu",
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        weight_init_scale_strategy: Optional[str] = "depthwise",
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
        conv_activation: str = "silu",
    ):
        weight_init_gains = _get_weight_init_gains(weight_init_scale_strategy, num_layers)
        emformer_layers = torch.nn.ModuleList(
            [
                _ConvEmformerLayer(
                    input_dim,
                    num_heads,
                    ffn_dim,
                    segment_length,
                    kernel_size,
                    dropout=dropout,
                    ffn_activation=ffn_activation,
                    left_context_length=left_context_length,
                    right_context_length=right_context_length,
                    max_memory_size=max_memory_size,
                    weight_init_gain=weight_init_gains[layer_idx],
                    tanh_on_mem=tanh_on_mem,
                    negative_inf=negative_inf,
                    conv_activation=conv_activation,
                )
                for layer_idx in range(num_layers)
            ]
        )
        super().__init__(
            emformer_layers,
            segment_length,
            left_context_length=left_context_length,
            right_context_length=right_context_length,
            max_memory_size=max_memory_size,
        )
