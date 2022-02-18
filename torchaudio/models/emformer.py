import math
from typing import List, Optional, Tuple

import torch


__all__ = ["Emformer"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


def _gen_padding_mask(
    utterance: torch.Tensor,
    right_context: torch.Tensor,
    summary: torch.Tensor,
    lengths: torch.Tensor,
    mems: torch.Tensor,
    left_context_key: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    T = right_context.size(0) + utterance.size(0) + summary.size(0)
    B = right_context.size(1)
    if B == 1:
        padding_mask = None
    else:
        right_context_blocks_length = T - torch.max(lengths).int() - summary.size(0)
        left_context_blocks_length = left_context_key.size(0) if left_context_key is not None else 0
        klengths = lengths + mems.size(0) + right_context_blocks_length + left_context_blocks_length
        padding_mask = _lengths_to_padding_mask(lengths=klengths)
    return padding_mask


def _get_activation_module(activation: str) -> torch.nn.Module:
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "gelu":
        return torch.nn.GELU()
    elif activation == "silu":
        return torch.nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation {activation}")


def _get_weight_init_gains(weight_init_scale_strategy: Optional[str], num_layers: int) -> List[Optional[float]]:
    if weight_init_scale_strategy is None:
        return [None for _ in range(num_layers)]
    elif weight_init_scale_strategy == "depthwise":
        return [1.0 / math.sqrt(layer_idx + 1) for layer_idx in range(num_layers)]
    elif weight_init_scale_strategy == "constant":
        return [1.0 / math.sqrt(2) for layer_idx in range(num_layers)]
    else:
        raise ValueError(f"Unsupported weight_init_scale_strategy value {weight_init_scale_strategy}")


def _gen_attention_mask_block(
    col_widths: List[int], col_mask: List[bool], num_rows: int, device: torch.device
) -> torch.Tensor:
    assert len(col_widths) == len(col_mask), "Length of col_widths must match that of col_mask"

    mask_block = [
        torch.ones(num_rows, col_width, device=device)
        if is_ones_col
        else torch.zeros(num_rows, col_width, device=device)
        for col_width, is_ones_col in zip(col_widths, col_mask)
    ]
    return torch.cat(mask_block, dim=1)


class _EmformerAttention(torch.nn.Module):
    r"""Emformer layer attention module.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Emformer layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        weight_init_gain (float or None, optional): scale factor to apply when initializing
            attention module parameters. (Default: ``None``)
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        weight_init_gain: Optional[float] = None,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) is not a multiple of num_heads ({num_heads}).")

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.tanh_on_mem = tanh_on_mem
        self.negative_inf = negative_inf

        self.scaling = (self.input_dim // self.num_heads) ** -0.5

        self.emb_to_key_value = torch.nn.Linear(input_dim, 2 * input_dim, bias=True)
        self.emb_to_query = torch.nn.Linear(input_dim, input_dim, bias=True)
        self.out_proj = torch.nn.Linear(input_dim, input_dim, bias=True)

        if weight_init_gain:
            torch.nn.init.xavier_uniform_(self.emb_to_key_value.weight, gain=weight_init_gain)
            torch.nn.init.xavier_uniform_(self.emb_to_query.weight, gain=weight_init_gain)

    def _gen_key_value(self, input: torch.Tensor, mems: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T, _, _ = input.shape
        summary_length = mems.size(0) + 1
        right_ctx_utterance_block = input[: T - summary_length]
        mems_right_ctx_utterance_block = torch.cat([mems, right_ctx_utterance_block])
        key, value = self.emb_to_key_value(mems_right_ctx_utterance_block).chunk(chunks=2, dim=2)
        return key, value

    def _gen_attention_probs(
        self,
        attention_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attention_weights_float = attention_weights.float()
        attention_weights_float = attention_weights_float.masked_fill(attention_mask.unsqueeze(0), self.negative_inf)
        T = attention_weights.size(1)
        B = attention_weights.size(0) // self.num_heads
        if padding_mask is not None:
            attention_weights_float = attention_weights_float.view(B, self.num_heads, T, -1)
            attention_weights_float = attention_weights_float.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), self.negative_inf
            )
            attention_weights_float = attention_weights_float.view(B * self.num_heads, T, -1)
        attention_probs = torch.nn.functional.softmax(attention_weights_float, dim=-1).type_as(attention_weights)
        return torch.nn.functional.dropout(attention_probs, p=float(self.dropout), training=self.training)

    def _forward_impl(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_val: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = utterance.size(1)
        T = right_context.size(0) + utterance.size(0) + summary.size(0)

        # Compute query with [right context, utterance, summary].
        query = self.emb_to_query(torch.cat([right_context, utterance, summary]))

        # Compute key and value with [mems, right context, utterance].
        key, value = self.emb_to_key_value(torch.cat([mems, right_context, utterance])).chunk(chunks=2, dim=2)

        if left_context_key is not None and left_context_val is not None:
            right_context_blocks_length = T - torch.max(lengths).int() - summary.size(0)
            key = torch.cat(
                [
                    key[: mems.size(0) + right_context_blocks_length],
                    left_context_key,
                    key[mems.size(0) + right_context_blocks_length :],
                ],
            )
            value = torch.cat(
                [
                    value[: mems.size(0) + right_context_blocks_length],
                    left_context_val,
                    value[mems.size(0) + right_context_blocks_length :],
                ],
            )

        # Compute attention weights from query, key, and value.
        reshaped_query, reshaped_key, reshaped_value = [
            tensor.contiguous().view(-1, B * self.num_heads, self.input_dim // self.num_heads).transpose(0, 1)
            for tensor in [query, key, value]
        ]
        attention_weights = torch.bmm(reshaped_query * self.scaling, reshaped_key.transpose(1, 2))

        # Compute padding mask.
        padding_mask = _gen_padding_mask(utterance, right_context, summary, lengths, mems, left_context_key)

        # Compute attention probabilities.
        attention_probs = self._gen_attention_probs(attention_weights, attention_mask, padding_mask)

        # Compute attention.
        attention = torch.bmm(attention_probs, reshaped_value)
        assert attention.shape == (
            B * self.num_heads,
            T,
            self.input_dim // self.num_heads,
        )
        attention = attention.transpose(0, 1).contiguous().view(T, B, self.input_dim)

        # Apply output projection.
        output_right_context_mems = self.out_proj(attention)

        summary_length = summary.size(0)
        output_right_context = output_right_context_mems[: T - summary_length]
        output_mems = output_right_context_mems[T - summary_length :]
        if self.tanh_on_mem:
            output_mems = torch.tanh(output_mems)
        else:
            output_mems = torch.clamp(output_mems, min=-10, max=10)

        return output_right_context, output_mems, key, value

    def forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        S: number of summary elements;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            summary (torch.Tensor): summary elements, with shape `(S, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            attention_mask (torch.Tensor): attention mask for underlying attention module.

        Returns:
            (Tensor, Tensor):
                Tensor
                    output frames corresponding to utterance and right_context, with shape `(T + R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
        """
        output, output_mems, _, _ = self._forward_impl(utterance, lengths, right_context, summary, mems, attention_mask)
        return output, output_mems[:-1]

    @torch.jit.export
    def infer(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        summary: torch.Tensor,
        mems: torch.Tensor,
        left_context_key: torch.Tensor,
        left_context_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for inference.

        B: batch size;
        D: feature dimension of each frame;
        T: number of utterance frames;
        R: number of right context frames;
        S: number of summary elements;
        M: number of memory elements.

        Args:
            utterance (torch.Tensor): utterance frames, with shape `(T, B, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``utterance``.
            right_context (torch.Tensor): right context frames, with shape `(R, B, D)`.
            summary (torch.Tensor): summary elements, with shape `(S, B, D)`.
            mems (torch.Tensor): memory elements, with shape `(M, B, D)`.
            left_context_key (torch.Tensor): left context attention key computed from preceding invocation.
            left_context_val (torch.Tensor): left context attention value computed from preceding invocation.

        Returns:
            (Tensor, Tensor, Tensor, and Tensor):
                Tensor
                    output frames corresponding to utterance and right_context, with shape `(T + R, B, D)`.
                Tensor
                    updated memory elements, with shape `(M, B, D)`.
                Tensor
                    attention key computed for left context and utterance.
                Tensor
                    attention value computed for left context and utterance.
        """
        query_dim = right_context.size(0) + utterance.size(0) + summary.size(0)
        key_dim = right_context.size(0) + utterance.size(0) + mems.size(0) + left_context_key.size(0)
        attention_mask = torch.zeros(query_dim, key_dim).to(dtype=torch.bool, device=utterance.device)
        attention_mask[-1, : mems.size(0)] = True
        output, output_mems, key, value = self._forward_impl(
            utterance,
            lengths,
            right_context,
            summary,
            mems,
            attention_mask,
            left_context_key=left_context_key,
            left_context_val=left_context_val,
        )
        return (
            output,
            output_mems,
            key[mems.size(0) + right_context.size(0) :],
            value[mems.size(0) + right_context.size(0) :],
        )


class _EmformerLayer(torch.nn.Module):
    r"""Emformer layer that constitutes Emformer.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads.
        ffn_dim: (int): hidden layer dimension of feedforward network.
        segment_length (int): length of each input segment.
        dropout (float, optional): dropout probability. (Default: 0.0)
        activation (str, optional): activation function to use in feedforward network.
            Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        weight_init_gain (float or None, optional): scale factor to apply when initializing
            attention module parameters. (Default: ``None``)
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        segment_length: int,
        dropout: float = 0.0,
        activation: str = "relu",
        left_context_length: int = 0,
        max_memory_size: int = 0,
        weight_init_gain: Optional[float] = None,
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

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

        activation_module = _get_activation_module(activation)
        self.pos_ff = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, ffn_dim),
            activation_module,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, input_dim),
            torch.nn.Dropout(dropout),
        )
        self.layer_norm_input = torch.nn.LayerNorm(input_dim)
        self.layer_norm_output = torch.nn.LayerNorm(input_dim)

        self.left_context_length = left_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size
        self.input_dim = input_dim

        self.use_mem = max_memory_size > 0

    def _init_state(self, batch_size: int, device: Optional[torch.device]) -> List[torch.Tensor]:
        empty_memory = torch.zeros(self.max_memory_size, batch_size, self.input_dim, device=device)
        left_context_key = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        left_context_val = torch.zeros(self.left_context_length, batch_size, self.input_dim, device=device)
        past_length = torch.zeros(1, batch_size, dtype=torch.int32, device=device)
        return [empty_memory, left_context_key, left_context_val, past_length]

    def _unpack_state(self, state: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        past_length = state[3][0][0].item()
        past_left_context_length = min(self.left_context_length, past_length)
        past_mem_length = min(self.max_memory_size, math.ceil(past_length / self.segment_length))
        pre_mems = state[0][self.max_memory_size - past_mem_length :]
        lc_key = state[1][self.left_context_length - past_left_context_length :]
        lc_val = state[2][self.left_context_length - past_left_context_length :]
        return pre_mems, lc_key, lc_val

    def _pack_state(
        self,
        next_k: torch.Tensor,
        next_v: torch.Tensor,
        update_length: int,
        mems: torch.Tensor,
        state: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        new_k = torch.cat([state[1], next_k])
        new_v = torch.cat([state[2], next_v])
        state[0] = torch.cat([state[0], mems])[-self.max_memory_size :]
        state[1] = new_k[new_k.shape[0] - self.left_context_length :]
        state[2] = new_v[new_v.shape[0] - self.left_context_length :]
        state[3] = state[3] + update_length
        return state

    def _process_attention_output(
        self,
        rc_output: torch.Tensor,
        utterance: torch.Tensor,
        right_context: torch.Tensor,
    ) -> torch.Tensor:
        result = self.dropout(rc_output) + torch.cat([right_context, utterance])
        result = self.pos_ff(result) + result
        result = self.layer_norm_output(result)
        return result

    def _apply_pre_attention_layer_norm(
        self, utterance: torch.Tensor, right_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer_norm_input = self.layer_norm_input(torch.cat([right_context, utterance]))
        return (
            layer_norm_input[right_context.size(0) :],
            layer_norm_input[: right_context.size(0)],
        )

    def _apply_post_attention_ffn(
        self, rc_output: torch.Tensor, utterance: torch.Tensor, right_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rc_output = self._process_attention_output(rc_output, utterance, right_context)
        return rc_output[right_context.size(0) :], rc_output[: right_context.size(0)]

    def _apply_attention_forward(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            raise ValueError("attention_mask must be not None when for_inference is False")

        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        rc_output, next_m = self.attention(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=summary,
            mems=mems,
            attention_mask=attention_mask,
        )
        return rc_output, next_m

    def _apply_attention_infer(
        self,
        utterance: torch.Tensor,
        lengths: torch.Tensor,
        right_context: torch.Tensor,
        mems: torch.Tensor,
        state: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if state is None:
            state = self._init_state(utterance.size(1), device=utterance.device)
        pre_mems, lc_key, lc_val = self._unpack_state(state)
        if self.use_mem:
            summary = self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
            summary = summary[:1]
        else:
            summary = torch.empty(0).to(dtype=utterance.dtype, device=utterance.device)
        rc_output, next_m, next_k, next_v = self.attention.infer(
            utterance=utterance,
            lengths=lengths,
            right_context=right_context,
            summary=summary,
            mems=pre_mems,
            left_context_key=lc_key,
            left_context_val=lc_val,
        )
        state = self._pack_state(next_k, next_v, utterance.size(0), mems, state)
        return rc_output, next_m, state

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
        (
            layer_norm_utterance,
            layer_norm_right_context,
        ) = self._apply_pre_attention_layer_norm(utterance, right_context)
        rc_output, output_mems = self._apply_attention_forward(
            layer_norm_utterance,
            lengths,
            layer_norm_right_context,
            mems,
            attention_mask,
        )
        output_utterance, output_right_context = self._apply_post_attention_ffn(rc_output, utterance, right_context)
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
        (
            layer_norm_utterance,
            layer_norm_right_context,
        ) = self._apply_pre_attention_layer_norm(utterance, right_context)
        rc_output, output_mems, output_state = self._apply_attention_infer(
            layer_norm_utterance, lengths, layer_norm_right_context, mems, state
        )
        output_utterance, output_right_context = self._apply_post_attention_ffn(rc_output, utterance, right_context)
        return output_utterance, output_right_context, output_state, output_mems


class Emformer(torch.nn.Module):
    r"""Implements the Emformer architecture introduced in
    *Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition*
    [:footcite:`shi2021emformer`].

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Emformer layer.
        ffn_dim (int): hidden layer dimension of each Emformer layer's feedforward network.
        num_layers (int): number of Emformer layers to instantiate.
        segment_length (int): length of each input segment.
        dropout (float, optional): dropout probability. (Default: 0.0)
        activation (str, optional): activation function to use in each Emformer layer's
            feedforward network. Must be one of ("relu", "gelu", "silu"). (Default: "relu")
        left_context_length (int, optional): length of left context. (Default: 0)
        right_context_length (int, optional): length of right context. (Default: 0)
        max_memory_size (int, optional): maximum number of memory elements to use. (Default: 0)
        weight_init_scale_strategy (str, optional): per-layer weight initialization scaling
            strategy. Must be one of ("depthwise", "constant", ``None``). (Default: "depthwise")
        tanh_on_mem (bool, optional): if ``True``, applies tanh to memory elements. (Default: ``False``)
        negative_inf (float, optional): value to use for negative infinity in attention weights. (Default: -1e8)

    Examples:
        >>> emformer = Emformer(512, 8, 2048, 20, 4, right_context_length=1)
        >>> input = torch.rand(128, 400, 512)  # batch, num_frames, feature_dim
        >>> lengths = torch.randint(1, 200, (128,))  # batch
        >>> output = emformer(input, lengths)
        >>> input = torch.rand(128, 5, 512)
        >>> lengths = torch.ones(128) * 5
        >>> output, lengths, states = emformer.infer(input, lengths, None)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        segment_length: int,
        dropout: float = 0.0,
        activation: str = "relu",
        left_context_length: int = 0,
        right_context_length: int = 0,
        max_memory_size: int = 0,
        weight_init_scale_strategy: str = "depthwise",
        tanh_on_mem: bool = False,
        negative_inf: float = -1e8,
    ):
        super().__init__()

        self.use_mem = max_memory_size > 0
        self.memory_op = torch.nn.AvgPool1d(
            kernel_size=segment_length,
            stride=segment_length,
            ceil_mode=True,
        )

        weight_init_gains = _get_weight_init_gains(weight_init_scale_strategy, num_layers)
        self.emformer_layers = torch.nn.ModuleList(
            [
                _EmformerLayer(
                    input_dim,
                    num_heads,
                    ffn_dim,
                    segment_length,
                    dropout=dropout,
                    activation=activation,
                    left_context_length=left_context_length,
                    max_memory_size=max_memory_size,
                    weight_init_gain=weight_init_gains[layer_idx],
                    tanh_on_mem=tanh_on_mem,
                    negative_inf=negative_inf,
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.segment_length = segment_length
        self.max_memory_size = max_memory_size

    def _gen_right_context(self, input: torch.Tensor) -> torch.Tensor:
        T = input.shape[0]
        num_segs = math.ceil((T - self.right_context_length) / self.segment_length)
        right_context_blocks = []
        for seg_idx in range(num_segs - 1):
            start = (seg_idx + 1) * self.segment_length
            end = start + self.right_context_length
            right_context_blocks.append(input[start:end])
        right_context_blocks.append(input[T - self.right_context_length :])
        return torch.cat(right_context_blocks)

    def _gen_attention_mask_col_widths(self, seg_idx: int, utterance_length: int) -> List[int]:
        num_segs = math.ceil(utterance_length / self.segment_length)
        rc = self.right_context_length
        lc = self.left_context_length
        rc_start = seg_idx * rc
        rc_end = rc_start + rc
        seg_start = max(seg_idx * self.segment_length - lc, 0)
        seg_end = min((seg_idx + 1) * self.segment_length, utterance_length)
        rc_length = self.right_context_length * num_segs

        if self.use_mem:
            m_start = max(seg_idx - self.max_memory_size, 0)
            mem_length = num_segs - 1
            col_widths = [
                m_start,  # before memory
                seg_idx - m_start,  # memory
                mem_length - seg_idx,  # after memory
                rc_start,  # before right context
                rc,  # right context
                rc_length - rc_end,  # after right context
                seg_start,  # before query segment
                seg_end - seg_start,  # query segment
                utterance_length - seg_end,  # after query segment
            ]
        else:
            col_widths = [
                rc_start,  # before right context
                rc,  # right context
                rc_length - rc_end,  # after right context
                seg_start,  # before query segment
                seg_end - seg_start,  # query segment
                utterance_length - seg_end,  # after query segment
            ]

        return col_widths

    def _gen_attention_mask(self, input: torch.Tensor) -> torch.Tensor:
        utterance_length = input.size(0)
        num_segs = math.ceil(utterance_length / self.segment_length)

        rc_mask = []
        query_mask = []
        summary_mask = []

        if self.use_mem:
            num_cols = 9
            # memory, right context, query segment
            rc_q_cols_mask = [idx in [1, 4, 7] for idx in range(num_cols)]
            # right context, query segment
            s_cols_mask = [idx in [4, 7] for idx in range(num_cols)]
            masks_to_concat = [rc_mask, query_mask, summary_mask]
        else:
            num_cols = 6
            # right context, query segment
            rc_q_cols_mask = [idx in [1, 4] for idx in range(num_cols)]
            s_cols_mask = None
            masks_to_concat = [rc_mask, query_mask]

        for seg_idx in range(num_segs):
            col_widths = self._gen_attention_mask_col_widths(seg_idx, utterance_length)

            rc_mask_block = _gen_attention_mask_block(
                col_widths, rc_q_cols_mask, self.right_context_length, input.device
            )
            rc_mask.append(rc_mask_block)

            query_mask_block = _gen_attention_mask_block(
                col_widths,
                rc_q_cols_mask,
                min(
                    self.segment_length,
                    utterance_length - seg_idx * self.segment_length,
                ),
                input.device,
            )
            query_mask.append(query_mask_block)

            if s_cols_mask is not None:
                summary_mask_block = _gen_attention_mask_block(col_widths, s_cols_mask, 1, input.device)
                summary_mask.append(summary_mask_block)

        attention_mask = (1 - torch.cat([torch.cat(mask) for mask in masks_to_concat])).to(torch.bool)
        return attention_mask

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass for training and non-streaming inference.

        B: batch size;
        T: max number of input frames in batch;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): utterance frames right-padded with right context frames, with
                shape `(B, T + right_context_length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid utterance frames for i-th batch element in ``input``.

        Returns:
            (Tensor, Tensor):
                Tensor
                    output frames, with shape `(B, T, D)`.
                Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        input = input.permute(1, 0, 2)
        right_context = self._gen_right_context(input)
        utterance = input[: input.size(0) - self.right_context_length]
        attention_mask = self._gen_attention_mask(utterance)
        mems = (
            self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)[:-1]
            if self.use_mem
            else torch.empty(0).to(dtype=input.dtype, device=input.device)
        )
        output = utterance
        for layer in self.emformer_layers:
            output, right_context, mems = layer(output, lengths, right_context, mems, attention_mask)
        return output.permute(1, 0, 2), lengths

    @torch.jit.export
    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass for streaming inference.

        B: batch size;
        D: feature dimension of each frame.

        Args:
            input (torch.Tensor): utterance frames right-padded with right context frames, with
                shape `(B, segment_length + right_context_length, D)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            states (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing Emformer internal state generated in preceding invocation of ``infer``. (Default: ``None``)

        Returns:
            (Tensor, Tensor, List[List[Tensor]]):
                Tensor
                    output frames, with shape `(B, segment_length, D)`.
                Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
                List[List[Tensor]]
                    output states; list of lists of tensors representing Emformer internal state
                    generated in current invocation of ``infer``.
        """
        assert input.size(1) == self.segment_length + self.right_context_length, (
            "Per configured segment_length and right_context_length"
            f", expected size of {self.segment_length + self.right_context_length} for dimension 1 of input"
            f", but got {input.size(1)}."
        )
        input = input.permute(1, 0, 2)
        right_context_start_idx = input.size(0) - self.right_context_length
        right_context = input[right_context_start_idx:]
        utterance = input[:right_context_start_idx]
        output_lengths = torch.clamp(lengths - self.right_context_length, min=0)
        mems = (
            self.memory_op(utterance.permute(1, 2, 0)).permute(2, 0, 1)
            if self.use_mem
            else torch.empty(0).to(dtype=input.dtype, device=input.device)
        )
        output = utterance
        output_states: List[List[torch.Tensor]] = []
        for layer_idx, layer in enumerate(self.emformer_layers):
            output, right_context, output_state, mems = layer.infer(
                output,
                output_lengths,
                right_context,
                None if states is None else states[layer_idx],
                mems,
            )
            output_states.append(output_state)

        return output.permute(1, 0, 2), output_lengths, output_states
