import math
from typing import List, Optional, Tuple

import torch


__all__ = ["Conformer"]


PADDING_IDX = 1


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


def _make_positions(input, padding_idx: int):
    mask = input.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).to(mask) * mask).long() + padding_idx


def _get_sinusoidal_embeddings(
    num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
) -> torch.Tensor:
    r"""Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    half_dim = embedding_dim // 2
    t = (torch.arange(half_dim, dtype=torch.float) * -math.log(10000) / (half_dim - 1)).exp()
    embedding_t = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * t.unsqueeze(0)
    embeddings = torch.cat([embedding_t.sin(), embedding_t.cos()], dim=1)
    if embedding_dim % 2 == 1:
        embeddings = torch.cat([embeddings, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        embeddings[padding_idx, :] = 0
    return embeddings.to(dtype=torch.float32)


class ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert (depthwise_kernel_size - 1) % 2 == 0, "depthwise_kernel_size must be odd to achieve 'SAME' padding."
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.ffn1 = FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
        )

        self.ffn2 = FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        x = x.transpose(0, 1)
        x = self.conv_module(x)
        x = x.transpose(0, 1)
        x = residual + x

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conv1dSubsampler(torch.nn.Module):
    r"""Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): number of input channels.
        mid_channels (int): number of intermediate channels.
        out_channels (int): number of output channels.
        kernel_sizes (List[int]): kernel size for each convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int],
    ) -> None:
        super().__init__()
        self.num_layers = len(kernel_sizes)
        conv_glus = [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels if i == 0 else mid_channels // 2,
                    mid_channels if i < self.num_layers - 1 else out_channels * 2,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                ),
                torch.nn.GLU(dim=1),
            )
            for i, kernel_size in enumerate(kernel_sizes)
        ]
        self.sequential = torch.nn.Sequential(*conv_glus)

    def _get_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = lengths
        for _ in range(self.num_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out.to(torch.int32)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): input frames, with shape `(B, T_in, in_channels)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor):
                torch.Tensor
                    output frames, with shape `(B, T_out, out_channels)`.
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        x = input.transpose(1, 2).contiguous()
        x = self.sequential(x)
        x = x.transpose(1, 2).contiguous()
        return x, self._get_output_lengths(lengths)


class SinusoidalPositionalEmbedding(torch.nn.Module):
    r"""Produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored.

    Args:
        embedding_dim (int): embedding dimension.
        padding_idx (int, optional): index corresponding to last padding symbol. (Default: 0)
        init_size (int, optional): initial embedding count. (Default: 1024)
    """

    def __init__(self, embedding_dim: int, padding_idx: int = 0, init_size: int = 1024) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embeddings = _get_sinusoidal_embeddings(init_size, embedding_dim, padding_idx)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, embedding_dim)`.
        """
        B, T = input.shape
        max_pos = self.padding_idx + 1 + T
        if max_pos > self.embeddings.size(0):
            self.embeddings = _get_sinusoidal_embeddings(max_pos, self.embedding_dim, self.padding_idx)
        self.embeddings = self.embeddings.to(input)
        positions = _make_positions(input, self.padding_idx)
        return self.embeddings.index_select(0, positions.view(-1)).view(B, T, -1).detach()


class Conformer(torch.nn.Module):
    r"""Implements the Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    [:footcite:`gulati2020conformer`].

    Args:
        num_layers (int): number of Conformer layers to instantiate.
        input_dim (int): input dimension.
        conv_channels (int): number of intermediate convolutional subsampler channels.
        conformer_layer_input_dim (int): Conformer layer input dimension.
        conv_kernel_sizes (List[int]): convolutional subsampler kernel sizes.
        max_source_positions (int): maximum input length.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)

    Examples:
        >>> conformer = Conformer(
        >>>     num_layers=4,
        >>>     input_dim=80,
        >>>     conv_channels=64,
        >>>     conformer_layer_input_dim=256,
        >>>     conv_kernel_sizes=[5, 5],
        >>>     max_source_positions=1000,
        >>>     ffn_dim=128,
        >>>     num_attention_heads=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        num_layers: int,
        input_dim: int,
        conv_channels: int,
        conformer_layer_input_dim: int,
        conv_kernel_sizes: List[int],
        max_source_positions: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.subsample = Conv1dSubsampler(
            input_dim,
            conv_channels,
            conformer_layer_input_dim,
            conv_kernel_sizes,
        )
        self.position_embedding = SinusoidalPositionalEmbedding(
            conformer_layer_input_dim,
            padding_idx=PADDING_IDX,
            init_size=max_source_positions + PADDING_IDX + 1,
        )
        self.linear = torch.nn.Linear(conformer_layer_input_dim, conformer_layer_input_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    conformer_layer_input_dim,
                    ffn_dim,
                    num_attention_heads,
                    depthwise_conv_kernel_size,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T_in, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T_out, conformer_layer_input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        x, lengths = self.subsample(input, lengths)
        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        positions = self.position_embedding(encoder_padding_mask)
        x += positions
        x = self.linear(x)
        x = self.dropout(x)

        x = x.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths
