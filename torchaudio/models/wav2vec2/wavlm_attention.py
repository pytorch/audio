"""
The MIT License (MIT)

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


class WavLMSelfAttention(nn.Module):
    """Multi-headed self-attention for WavLM model :cite:`chen2022wavlm`.
    Wraps around ``torch.nn.MultiheadAttention``, creating relaive position embeddings and passing them to multi-headed
    attention as a mask.
    Source: https://github.com/microsoft/unilm/blob/2d8302f09c99bca2b82e6e868d81d4281cceebc8/wavlm/modules.py#L303-L763

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional): Dropout probability on attn_output_weights. (Default: to ``0.0``)
        bias (bool, optional): If ``True``, add bias to input / output projection layers. (Default: ``True``)
        has_relative_attention_bias (bool, optional): If ``True``, apply relative position embedding.
            Necessary in the first encoder layer, but not in the subsequent ones. (Default: ``False``)
        num_buckets (int, optional): Number of buckets for relative position embedding. (Default: ``32``)
        max_distance (int, optional): Naximum distance for relative position embedding. (Default: ``128``)
        gru_rel_pos (bool, optional): If ``True``, apply gated relative position embedding. (Default: ``False``)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        if has_relative_attention_bias:
            self.rel_attn_embed = nn.Embedding(num_buckets, num_heads)
        else:
            self.rel_attn_embed = None

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.dropout = dropout
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)

        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.gru_rel_pos_linear = nn.Linear(self.head_dim, 8)
            self.gru_rel_pos_const = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.has_position_bias = True

    def compute_bias(self, query_length: int, key_length: int) -> Tensor:
        """Compute relative position embeddings for WavLM model.
        Args:
            query_length (int): Query position can take values between 0 and ``query_length - 1``.
            key_length (int): Key position can take values between 0 and ``key_length - 1``.
        Returns:
            Tensor of shape `(num_heads, query_length, key_length)`, relative positions embeddings
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # Shape (query_length, key_length)
        relative_position_bucket = self._relative_positions_bucket(relative_position, bidirectional=True)
        relative_position_bucket = relative_position_bucket.to(self.rel_attn_embed.weight.device)
        values = self.rel_attn_embed(relative_position_bucket)  # Shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1])
        return values

    def _relative_positions_bucket(self, relative_positions: Tensor, bidirectional: bool = True):
        """Compute relative position buckets for WavLM model. Computation similar to formula (5) in WavLM
           paper :cite:`chen2022wavlm`.
        Args:
            relative_positions (Tensor): Relative offsets between query and key positions,
                of shape ``(query_length, key_length)``.
            bidirectional (bool): If ``True``, values will be filled both above and below the diagonal in the resulting
                matrix. If ``False``, the elements above the diagonal (i.e. with negative relative offsets) will be set
                to zero. (Default ``True``)
        Returns:
            Tensor of shape ``(query_length, key_length)`` filled bucketed values of with relative positions.
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        # Shape (query_length, key_length)
        relative_buckets = torch.zeros_like(relative_positions, dtype=torch.long)

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact

        relative_postion_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_positions, relative_postion_if_large)
        return relative_buckets

    def forward(
        self,
        query: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query (Tensor): Input of shape ``(batch_size, src_len, embed_dim)``.
            key_padding_mask (Tensor or None, optional): Mask to exclude keys that are pads, of shape
                `(batch, src_len)`, where padding elements are indicated by 1s. (Default: ``None``)
            attn_mask: Needs to be ``None``. The argument exists for compatibility with
                ``EncoderLayer``. (Default: ``None``)
            position_bias (Tensor or None, optional): Position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``. When used inside WavLM model encoder, will be
                generated in the first layer and then passed from each encoder layer to the next one.
                (Default: ``None``)
        Returns:
            attn_output (Tensor): Attention output of shape ``(batch_size, src_len, embed_dim)``.
            position_bias (Tensor or None): Position bias of shape ``(batch_size * num_heads, src_len, src_len)``.
        """
        bsz, seq_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert attention_mask is None

        if self.rel_attn_embed is not None and position_bias is None:
            position_bias = self.compute_bias(seq_len, seq_len)
            position_bias = position_bias.unsqueeze(0).repeat(bsz, 1, 1, 1)

        attn_mask_rel_pos: Optional[Tensor] = None
        if position_bias is not None:
            attn_mask_rel_pos = position_bias
            if self.gru_rel_pos:  # Apply gating on relative position bias
                query_layer = query.view(bsz, seq_len, self.num_heads, -1)
                query_layer = query_layer.permute(0, 2, 1, 3)

                gate_a, gate_b = torch.sigmoid(
                    self.gru_rel_pos_linear(query_layer).view(bsz, self.num_heads, seq_len, 2, 4).sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.gru_rel_pos_const - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(bsz, self.num_heads, -1, 1) * position_bias

            attn_mask_rel_pos = attn_mask_rel_pos.view((bsz, self.num_heads, seq_len, seq_len))

        if attn_mask_rel_pos is not None and key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
            key_padding_mask = torch.nn.functional._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=torch.nn.functional._none_or_dtype(attn_mask_rel_pos),
                other_name="",
                target_type=query.dtype,
            )
        if attn_mask_rel_pos is not None and key_padding_mask is not None:
            attn_mask_rel_pos = attn_mask_rel_pos + key_padding_mask
        query_projected = torch.nn.functional.linear(query, self.attention.in_proj_weight, self.attention.in_proj_bias)
        query, key, value = query_projected.chunk(3, -1)
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        query = query.view(shape).transpose(2, 1)  # (batch, num_heads, seq_len, head_dim)
        key = key.view(shape).transpose(2, 1)  # (batch, num_heads, seq_len, head_dim)
        value = value.view(shape).transpose(2, 1)  # (batch, num_heads, seq_len, head_dim)
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask_rel_pos,
            dropout_p=dropout,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, -1, self.num_heads * self.head_dim)
        attn_output = self.attention.out_proj(attn_output)
        return attn_output, position_bias
