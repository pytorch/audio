from typing import Literal, Optional, Tuple

import torch


__all__ = ["Conformer"]

_NormType = Literal["batch_norm", "group_norm", "layer_norm"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _TransposedLayerNorm(torch.nn.LayerNorm):
    r"""Layer norm over the channel dimension of ``(B, C, T)`` tensors."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.transpose(-2, -1)
        x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.transpose(-2, -1)


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
        norm_type (str or None, optional): normalization applied after the depthwise
            convolution; one of ``"batch_norm"``, ``"group_norm"``, or ``"layer_norm"``.
            ``"layer_norm"`` normalizes each time step independently, so its statistics are
            unaffected by padding. ``"batch_norm"`` and ``"group_norm"`` pool statistics over
            the time dimension, which includes padded time steps even when a ``key_padding_mask``
            is passed to :meth:`forward`. For fully padding-invariant behavior, pass a
            ``key_padding_mask`` together with ``norm_type="layer_norm"``. If ``None``, the
            normalization is selected by ``use_group_norm``. Mutually exclusive with
            ``use_group_norm=True``. (Default: ``None``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
        norm_type: Optional[_NormType] = None,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        if norm_type is not None and use_group_norm:
            raise ValueError("Only one of use_group_norm and norm_type can be specified.")
        if norm_type is None:
            norm_type = "group_norm" if use_group_norm else "batch_norm"

        norm: torch.nn.Module
        if norm_type == "batch_norm":
            norm = torch.nn.BatchNorm1d(num_channels)
        elif norm_type == "group_norm":
            norm = torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
        elif norm_type == "layer_norm":
            norm = _TransposedLayerNorm(num_channels)
        else:
            raise ValueError(
                f"norm_type must be one of 'batch_norm', 'group_norm', or 'layer_norm', but got {norm_type!r}."
            )

        self.layer_norm = torch.nn.LayerNorm(input_dim)
        # Even though forward() may iterate over the modules one by one (to inject masking),
        # they must stay in this Sequential so that state_dict keys (e.g. ``sequential.2.weight``)
        # remain compatible with checkpoints of previously trained models.
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
            norm,
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

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.
            key_padding_mask (torch.Tensor or None): boolean mask with shape `(B, T)` in which ``True``
                marks a padded time step to zero out before the depthwise convolution. (Default: ``None``)

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        if key_padding_mask is None:
            x = self.sequential(x)
        else:
            padding_mask = key_padding_mask.unsqueeze(1)  # (B, 1, T)
            for i, module in enumerate(self.sequential):
                if i == 2:
                    # Zero padded frames so the depthwise conv can't bleed them into valid ones.
                    x = x.masked_fill(padding_mask, 0.0)
                x = module(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
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
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
        conv_mask_padding (bool, optional): if ``True``, zero out padded positions in the
            convolution module before the depthwise convolution, using the ``key_padding_mask``
            passed to :meth:`forward`. This stops padding from leaking into valid frames
            through the convolution. Note that the module's normalization can still let
            padding affect valid frames; see ``conv_norm_type``. Disabled by default for backward
            compatibility with models trained without masking. (Default: ``False``)
        conv_norm_type (str or None, optional): normalization applied after the depthwise
            convolution in the convolution module; one of ``"batch_norm"``,
            ``"group_norm"``, or ``"layer_norm"``. ``"layer_norm"`` normalizes each time step
            independently, so its statistics are unaffected by padding. ``"batch_norm"`` and
            ``"group_norm"`` pool statistics over the time dimension, which includes padded time
            steps even when ``conv_mask_padding=True``. For fully padding-invariant behavior, use
            ``conv_mask_padding=True`` together with ``conv_norm_type="layer_norm"``. If ``None``,
            the normalization is selected by ``use_group_norm``. Mutually exclusive with
            ``use_group_norm=True``. (Default: ``None``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        conv_mask_padding: bool = False,
        conv_norm_type: Optional[_NormType] = None,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
            norm_type=conv_norm_type,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first
        self.conv_mask_padding = conv_mask_padding

    def _apply_convolution(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input, key_padding_mask if self.conv_mask_padding else None)
        input = input.transpose(0, 1)
        input = residual + input
        return input

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

        if self.convolution_first:
            x = self._apply_convolution(x, key_padding_mask)

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

        if not self.convolution_first:
            x = self._apply_convolution(x, key_padding_mask)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
        conv_mask_padding (bool, optional): if ``True``, zero out padded positions in each
            layer's convolution module before the depthwise convolution, using the padding
            mask derived from ``lengths``. This stops padding from leaking into valid frames
            through the convolution. Note that the module's normalization can still let
            padding affect valid frames; see ``conv_norm_type``. Disabled by default for backward
            compatibility with models trained without masking. (Default: ``False``)
        conv_norm_type (str or None, optional): normalization applied after the depthwise
            convolution in each layer's convolution module; one of ``"batch_norm"``,
            ``"group_norm"``, or ``"layer_norm"``. ``"layer_norm"`` normalizes each time step
            independently, so its statistics are unaffected by padding. ``"batch_norm"`` and
            ``"group_norm"`` pool statistics over the time dimension, which includes padded time
            steps even when ``conv_mask_padding=True``. For fully padding-invariant behavior, use
            ``conv_mask_padding=True`` together with ``conv_norm_type="layer_norm"``. If ``None``,
            the normalization is selected by ``use_group_norm``. Mutually exclusive with
            ``use_group_norm=True``. (Default: ``None``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        conv_mask_padding: bool = False,
        conv_norm_type: Optional[_NormType] = None,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                    conv_mask_padding=conv_mask_padding,
                    conv_norm_type=conv_norm_type,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths
