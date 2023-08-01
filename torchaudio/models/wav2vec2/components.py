import logging
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import Module, Parameter

from .wavlm_attention import WavLMSelfAttention

_LG = logging.getLogger(__name__)


def _init_transformer_params(module):
    """
    Initialize the weights of Transformer module in Wav2Vec2/HuBERT.

    If the module is ``nn.Linear``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``bias`` is set to ``True`` in the module, set ``bias`` to 0.

    If the module is ``nn.Embedding``, normalize the weight with mean 0 and standard deviation 0.02.
    If ``padding_idx`` is not None, set the weight of padding to 0.

    Note:
        Ths method corresponds to
        `init_bert_params
        <https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/transformer_sentence_encoder.py#L21>`__
        in the original ``fairseq`` implementation.
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose"""

    def forward(self, input: Tensor) -> Tensor:
        x = input.transpose(-2, -1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.transpose(-2, -1)
        return x


class ConvLayerBlock(Module):
    """Convolution unit of FeatureExtractor"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[Module],
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Shape: ``[batch, in_channels, in_frame]``.
            length (Tensor or None, optional): Shape ``[batch, ]``.
        Returns:
            Tensor: Shape ``[batch, out_channels, out_frames]``.
            Optional[Tensor]: Shape ``[batch, ]``.
        """
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = nn.functional.gelu(x)

        if length is not None:
            length = torch.div(length - self.kernel_size, self.stride, rounding_mode="floor") + 1
            # When input length is 0, the resulting length can be negative. So fix it here.
            length = torch.max(torch.zeros_like(length), length)
        return x, length


class FeatureExtractor(Module):
    """Extract features from audio

    Args:
        conv_layers (nn.ModuleList):
            convolution layers
    """

    def __init__(
        self,
        conv_layers: nn.ModuleList,
    ):
        super().__init__()
        self.conv_layers = conv_layers

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing a batch of audio,
                shape: ``[batch, time]``.
            length (Tensor or None, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            Tensor:
                The resulting feature, shape: ``[batch, frame, feature]``
            Optional[Tensor]:
                Valid length of each output sample. shape: ``[batch, ]``.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected the input Tensor to be 2D (batch, time). Found: {list(x.shape)}")

        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x, length = layer(x, length)  # (batch, feature, frame)
        x = x.transpose(1, 2)  # (batch, frame, feature)
        return x, length


class FeatureProjection(Module):
    """Layer that connects FeatureExtractor and Encoder

    Projects features to encoder dimension.

    Args:
        in_features (int): Input feature dim.
        out_features (int): Output feature dim.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(
            in_features,
            out_features,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor):
                Feature Tensor. shape: ``[batch, frame, in_feature]``
        Returns:
            Tensor: Projected features. ``[batch, frame, out_feature]``.
        """
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ConvolutionalPositionalEmbedding(Module):
    """Positional embedding which is placed at the beginning of Transformer.

    Args:
        embed_dim (int): Feature dimension of the input Tensor.
        kernel_size (int): The number of frames to be use.
        groups (int): The number of groups in feature dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        groups: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )

        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.num_remove: int = 1 if kernel_size % 2 == 0 else 0

    def __prepare_scriptable__(self):
        if self.conv.__class__.__name__ == "ParametrizedConv1d":
            _LG.warning("Removing weight_norm from %s", self.__class__.__name__)
            torch.nn.utils.parametrize.remove_parametrizations(self.conv, "weight")
        return self

    def forward(self, x):
        """
        Args:
            x (Tensor): shape ``[batch, frame, feature]``.

        Returns:
            Tensor: The resulting feature. Shape ``[batch, frame, feature]``.
        """
        x = x.transpose(-2, -1)
        x = self.conv(x)
        if self.num_remove > 0:
            x = x[..., : -self.num_remove]
        x = torch.nn.functional.gelu(x)
        x = x.transpose(-2, -1)
        return x


class SelfAttention(Module):
    """Multihead Self Attention module

    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probability on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or ``None``, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
            position_bias: Not used. Only for the compatibility with :py:class:`WavLMSelfAttention`.
            key_padding_mask (Tensor or ``None``): Not used. Only for the compatibility with
                :py:class:`WavLMSelfAttention`.
        Returns:
            (Tensor, ``None``): The resulting attention output and ``None`` (necessary for compatibility
                with :py:class:`WavLMSelAttention`).
                Attention output shape: ``[batch, sequence_length, embed_dim]``.
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        dropout = self.dropout if self.training else 0.0
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(attn_output)
        return output, None  # Necessary for compatibility with WavLMSelAttention


class FeedForward(Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(
        self,
        io_features: int,
        intermediate_features: int,
        intermediate_dropout: float,
        output_dropout: float,
    ):
        super().__init__()
        self.intermediate_dense = nn.Linear(io_features, intermediate_features)
        self.intermediate_dropout = nn.Dropout(intermediate_dropout)
        self.output_dense = nn.Linear(intermediate_features, io_features)
        self.output_dropout = nn.Dropout(output_dropout)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        Returns:
            x (Tensor): shape: `(batch, sequence_length, io_features)`
        """
        x = self.intermediate_dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class EncoderLayer(Module):
    """A layer unit in encoder. Combines multihead self attention and feed forward."""

    def __init__(
        self,
        attention: Module,
        dropout: float,
        layer_norm_first: bool,
        feed_forward: Module,
    ):
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Input of shape ``(batch, sequence_length, embed_dim)``.
            attention_mask (Tensor or ``None``, optional): attention mask
                of shape ``(batch, 1, sequence_length, sequence_length)``. (Default: ``None``)
            position_bias (Tensor or ``None``, optional): position bias of shape
                ``(batch_size * num_heads, src_len, src_len)``.
                Only necessary for WavLM model, ``None`` otherwise. (Default: ``None``)
            key_padding_mask (Tensor or ``None``, optional): key padding mask of shape ``(batch_size, src_len)``.
                Only used for WavLM model, ignored otherwise. (Default: ``None``)
        Returns:
            (x, position_bias): Shapes are the same as in the input. Position bias is only relevant for WaLM model,
                ``None`` otherwise.
        """
        residual = x

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x, position_bias = self.attention(
            x, attention_mask=attention_mask, position_bias=position_bias, key_padding_mask=key_padding_mask
        )

        x = self.dropout(x)
        x = residual + x

        if self.layer_norm_first:
            x = x + self.feed_forward(self.final_layer_norm(x))
        else:
            x = self.layer_norm(x)
            x = self.final_layer_norm(x + self.feed_forward(x))
        return x, position_bias


class Transformer(Module):
    def __init__(
        self,
        pos_conv_embed: Module,
        dropout: float,
        layers: Module,
        layer_norm_first: bool,
        layer_drop: float,
    ):
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers

    def _preprocess(self, x: Tensor):
        x = x + self.pos_conv_embed(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._preprocess(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1).item() <= self.layer_drop):
                x, position_bias = layer(x, attention_mask, position_bias=position_bias)

        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.layers):
                raise ValueError(f"`num_layers` must be between [1, {len(self.layers)}]")

        ret: List[Tensor] = []
        position_bias = None
        x = self._preprocess(x)
        for layer in self.layers:
            x, position_bias = layer(x, attention_mask, position_bias=position_bias)
            ret.append(x)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret


class Encoder(Module):
    def __init__(
        self,
        feature_projection: Module,
        transformer: Module,
    ):
        super().__init__()
        self.feature_projection = feature_projection
        self.transformer = transformer

    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)

        mask: Optional[Tensor] = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            # create mask for padded elements and zero-out them
            mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
            x[mask] = 0.0
            # extend the mask to attention shape and set weight
            mask = -10000.0 * mask[:, None, None, :].to(dtype=features.dtype)
            mask = mask.expand(batch_size, 1, max_len, max_len)
        return x, mask

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        x, mask = self._preprocess(features, lengths)
        x = self.transformer(x, attention_mask=mask)
        return x

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        return self.transformer.get_intermediate_outputs(x, attention_mask=masks, num_layers=num_layers)


################################################################################
def _get_feature_extractor(
    norm_mode: str,
    shapes: List[Tuple[int, int, int]],
    bias: bool,
) -> FeatureExtractor:
    """
    Args:
        norm_mode (str):
            Either "group_norm" or "layer_norm".
            If "group_norm", then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.
            This option corresponds to "extractor_mode" from fairseq.
            Expected values are "group_norm" for Base arch, and
            "layer_norm" for Large arch.
        shapes (list of tuple of int):
            Configuration of convolution layers. List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``
            This option corresponds to "conv_feature_layers" from fairseq.
            Expected values are
            ``[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2``
            for all the architectures.
        bias (bool):
            Whether to include bias term to each convolution operation.
            This option corresponds to "conv_bias" from fairseq.
            Expected values are False for Base arch, and True for Large arch.

    See Also:
        * Original implementation
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L666-L733
        * "extractor_mode"
          - Def and base:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L38-L45
          - Large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L52
        * "conv_feature_layers"
          - Def, base and large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L94-L100
        * "conv_bias"
          - Def and base:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L101-L103
          - Large:
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L61
    """
    if norm_mode not in ["group_norm", "layer_norm"]:
        raise ValueError("Invalid norm mode")
    blocks = []
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(shapes):
        normalization = None
        if norm_mode == "group_norm" and i == 0:
            normalization = nn.GroupNorm(
                num_groups=out_channels,
                num_channels=out_channels,
                affine=True,
            )
        elif norm_mode == "layer_norm":
            normalization = LayerNorm(
                normalized_shape=out_channels,
                elementwise_affine=True,
            )
        blocks.append(
            ConvLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                layer_norm=normalization,
            )
        )
        in_channels = out_channels
    return FeatureExtractor(nn.ModuleList(blocks))


def _get_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    num_heads: int,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
) -> Encoder:
    """
    Args:
        in_features (int): The number of input features.
        embed_dim (int):
            The dimension of embedding.
            This option corresponds to "encoder_embed_dim" from fairseq.
            Expected values are 768 for Base arch, and 1024 for Large arch.
        dropout_input (float):
            The dropout probability applied after the input feature is projected
            to ``embed_dim``.
            This option corresponds to "dropout_input" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.
            This option corresponds to "conv_pos" from fairseq.
            Expected values are 128 for both Base and Large arch.
        pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.
            This option corresponds to "conv_pos_groups" from fairseq.
            Expected values are 16 for both Base and Large arch.
        num_layers (int):
            The number of self attention layers in transformer block.
            This option corresponds to "encoder_layers" from fairseq.
            Expected values are 12 for Base and 24 for Large arch.
        num_heads (int):
            The number of heads in self attention layers.
            This option corresponds to "encoder_attention_heads" from fairseq.
            Expected values are 12 for Base and 16 for Large arch.
        attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.
            This option corresponds to "attention_dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        ff_interm_features (int):
            The dimension of hidden features in feed forward layer.
            This option corresponds to "encoder_ffn_embed_dim" from fairseq.
            Expected values are 3072 for Base and 4096 for Large arch.
        ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.
            This option correspinds to "activation_dropout" from fairseq.
            Expected values are 0.1 for both Base and Large arch.
        dropout (float):
            The dropout probability applied at the end of feed forward layer.
            This option corresponds to "dropout" from fairseq.
            Expected values are 0.1 for Base and 0.0 for Large arch.
        layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.
            This option corresponds to "layer_norm_first" from fairseq.
            Expected values are False for Base and True for Large arch.
        layer_drop (float):
            Probability to drop each encoder layer during training.
            This option corresponds to "layerdrop" from fairseq.
            Expected values are 0.1 for both Base and Large arch.

    See Also:
        * "encoder_embed_dim"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L49-L51
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L64
        * "dropout_input"
          - Def, base and large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L75-L78
        * "conv_pos"
          - Def, base and large
            NOTE: The description is wrong.
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L204-L207
          - Usage
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L756
        * "conv_pos_groups"
          - Def, base and large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L208-L211
        * "encoder_layers"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L46-L48
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L63
        * "encoder_attention_heads"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L55-L57
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L66
        * "attention_dropout"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L66-L68
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L60
        * "encoder_ffn_embed_dim"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L52-L54
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L65
        * "activation_dropout"
          - Def
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L69-L71
          - Base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L55
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L55
        * "dropout"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L63-L65
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L59
        * "layer_norm_first"
          - Def and base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L91-L93
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml#L53
        * "layerdrop"
          - Def
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L72-L74
          - Base
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L54
          - Large
            https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L54
    """
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # Original impl
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    encoder_layers = nn.ModuleList()
    for _ in range(num_layers):
        attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        feed_forward = FeedForward(
            io_features=embed_dim,
            intermediate_features=ff_interm_features,
            intermediate_dropout=ff_interm_dropout,
            output_dropout=dropout,
        )
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


def _get_wavlm_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    pos_conv_kernel: int,
    pos_conv_groups: int,
    num_layers: int,
    num_heads: int,
    num_buckets: int,
    max_distance: int,
    attention_dropout: float,
    ff_interm_features: int,
    ff_interm_dropout: float,
    dropout: float,
    layer_norm_first: bool,
    layer_drop: float,
) -> Encoder:
    """
    Construct encoder for WavLM model :cite:`chen2022wavlm`. The structure of the encoder and most of the argments are
    the same as in :py:func:`_get_encoder` so refer there for documentation. The only difference from Wav2Vec2 encoder
    is usage of `WavLMSelfAttention` instead of `SelfAttention` and two additional parameters: `num_buckets` and
    `max_distance`.
    Args:
        in_features (int): See :py:func:`_get_encoder`.
        embed_dim (int): See :py:func:`_get_encoder`.
        dropout_input (float): See :py:func:`_get_encoder`.
        pos_conv_kernel (int): See :py:func:`_get_encoder`.
        pos_conv_groups (int): See :py:func:`_get_encoder`.
        num_layers (int): See :py:func:`_get_encoder`.
        num_heads (int): See :py:func:`_get_encoder`.
        num_buckets (int): Number of buckets for relative position embedding.
        max_distance (int): Maximum distance for relative position embedding.
        attention_dropout (float): See :py:func:`_get_encoder`.
        ff_interm_features (int): See :py:func:`_get_encoder`.
        ff_interm_dropout (float): See :py:func:`_get_encoder`.
        dropout (float): See :py:func:`_get_encoder`.
        layer_norm_first (bool): See :py:func:`_get_encoder`.
        layer_drop (float): See :py:func:`_get_encoder`.

    """
    feature_projection = FeatureProjection(in_features, embed_dim, dropout_input)
    pos_conv = ConvolutionalPositionalEmbedding(embed_dim, pos_conv_kernel, pos_conv_groups)

    # Original impl
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L768-L782
    encoder_layers = nn.ModuleList()
    for i in range(num_layers):
        attention = WavLMSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_buckets=num_buckets,
            max_distance=max_distance,
            dropout=attention_dropout,
            has_relative_attention_bias=(i == 0),  # Position embedding is only necessary in the first layer.
        )
        feed_forward = FeedForward(
            io_features=embed_dim,
            intermediate_features=ff_interm_features,
            intermediate_dropout=ff_interm_dropout,
            output_dropout=dropout,
        )
        encoder_layers.append(
            EncoderLayer(
                attention=attention,
                dropout=dropout,
                layer_norm_first=layer_norm_first,
                feed_forward=feed_forward,
            )
        )
    transformer = Transformer(
        pos_conv_embed=pos_conv,
        dropout=dropout,
        layers=encoder_layers,
        layer_norm_first=not layer_norm_first,
        layer_drop=layer_drop,
    )
    return Encoder(feature_projection, transformer)


def _compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> Tensor:
    """Computes random mask spans for a given shape.
    Args:
        shape (int, int): The shape for which to compute masks.
            The first element is batch size and second is the number of frames.
        padding_mask (Tensor or None): The padding mask of the same dimension as shape,
            which will prevent masking padded elements.
        mask_prob (float): Probability for each token to be chosen as start of the span to be masked.
            This will be multiplied by number of timesteps divided by length of mask span to mask
            approximately this percentage of all elements. However due to overlaps, the actual number
            will be smaller (unless no_overlap is True).
        mask_type (str): How to compute mask lengths. Options: [``static``, ``uniform``, ``normal``, ``poisson``].
            ``static``: Fixed size
            ``uniform``: Sample from uniform distribution [mask_other, mask_length*2]
            ``normal``: Sample from normal distribution with mean ``mask_length`` and stdev ``mask_other``.
            ``poisson``: Sample from possion distribution with lambda = ``mask_length``.
        min_masks (int): Minimum number of masked spans.
        no_overlap (bool): If false, will switch to an alternative recursive algorithm
            that prevents spans from overlapping.
        min_space (int): How many frames to keep unmasked between spans (Only used if no_overlap is True).

    Returns:
        (Tensor): The mask indices of dimension `[batch, frame]`.
    """

    batch_size, frame = shape
    mask = torch.full((batch_size, frame), False)
    # add a random number for probabilistic rounding
    all_num_mask = int(mask_prob * frame / float(mask_length) + torch.rand(1))

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(batch_size):
        if padding_mask is not None:
            sz = frame - padding_mask[i].long().sum().item()
            # add a random number for probabilistic rounding
            num_mask = int(mask_prob * sz / float(mask_length) + torch.rand(1))
            num_mask = max(min_masks, num_mask)
        else:
            sz = frame
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = torch.full((num_mask,), mask_length)
        elif mask_type == "uniform":
            lengths = torch.randint(mask_other, mask_length * 2 + 1, size=(num_mask,))
        elif mask_type == "normal":
            lengths = torch.normal(mask_length, mask_other, size=(num_mask,))
            lengths = torch.maximum(torch.ones(1), torch.round(lengths)).int()
        elif mask_type == "poisson":
            lengths = torch.poisson(mask_length, size=(num_mask,))
            lengths = torch.round(lengths).int()
        else:
            raise Exception(f"unknown mask selection: {mask_type}")

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = torch.randint(s, e - length, size=(1,))
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = torch.tensor([e - s for s, e in parts], dtype=torch.int)
                lens[lens < length + min_space] = 0
                l_sum = lens.sum()
                if l_sum == 0:
                    break
                probs = lens / l_sum
                c = torch.distributions.categorical.Categorical(probs).sample()
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = torch.tensor(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = torch.randperm(sz - min_len)[:num_mask]
            mask_idc = torch.tensor(
                [mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])]
            )

        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = mask_idc[torch.randperm(len(mask_idc))[:min_len].long()]
        mask[i, mask_idc] = True

    return mask


def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        input (Tensor): The padded Tensor of dimension `[batch, max_len, frequency]`.
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    batch_size, max_len, _ = input.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    return mask


class MaskGenerator(Module):
    """Generate the masks for masked prediction.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        mask_prob (float): Probability for each token to be chosen as start of the span to be masked.
            This will be multiplied by number of timesteps divided by length of mask span to mask
            approximately this percentage of all elements. However due to overlaps, the actual number
            will be smaller (unless no_overlap is True).
        mask_selection (str): How to choose the mask length.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_other (float): Secondary mask argument (used for more complex distributions).
        mask_length (int): The lengths of the mask.
        no_mask_overlap (bool):  Whether to allow masks to overlap.
        mask_min_space (int):  Minimum space between spans (if no overlap is enabled).
        mask_channel_prob (float): The probability of replacing a feature with 0.
        mask_channel_selection (str): How to choose the mask length for channel masking.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_channel_other (float): Secondary mask argument for channel masking(used for more complex distributions).
        mask_channel_length (int): Minimum space between spans (if no overlap is enabled) for channel masking.
        no_mask_channel_overlap (bool):  Whether to allow channel masks to overlap.
        mask_channel_min_space (int): Minimum space between spans for channel masking(if no overlap is enabled).
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        mask_prob: float,
        mask_selection: str,
        mask_other: float,
        mask_length: int,
        no_mask_overlap: bool,
        mask_min_space: int,
        mask_channel_prob: float,
        mask_channel_selection: str,
        mask_channel_other: float,
        mask_channel_length: int,
        no_mask_channel_overlap: bool,
        mask_channel_min_space: int,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.mask_length = mask_length
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.mask_channel_length = mask_channel_length
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.mask_embedding = Parameter(torch.FloatTensor(encoder_embed_dim))
        torch.nn.init.uniform_(self.mask_embedding)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            x (Tensor): The encoded representations after feature extraction module.
            padding_mask (Tensor or None): The padding mask of the same dimension as shape,
                which will prevent masking padded elements.

        Returns:
            Tensor: The feature representations after masking.
            Tensor: The generated mask indices.
        """
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = _compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = mask_indices.to(x.device)
            # change dtype of mask_embedding to x for mixed-precision training.
            # see https://github.com/pytorch/audio/issues/2847 for details.
            x[mask_indices] = self.mask_embedding.to(x.dtype)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = _compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = mask_channel_indices.to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        return x, mask_indices


def _compute_logits(
    proj_x: Tensor,
    target: Tensor,
    label_embeddings: Parameter,
) -> Tensor:
    """Compute the logits of the embeddings.
    Args:
        proj_x (Tensor): The projected masked representations of dimension `[batch, frame, final_dim]`.
        target (Tensor): The target Tensor of dimension `[batch, frame, final_dim]`.
        label_embeddings (Parameter): The trainable embeddings of target of dimension `[num_class, final_dim]`.

    Returns:
        (Tensor): The logits of the inputs.
    """
    logit_temp = 0.1
    pos = torch.index_select(label_embeddings, 0, target.long())
    negs = label_embeddings.unsqueeze(1).expand(-1, proj_x.size(0), -1)
    neg_is_pos = (pos == negs).all(-1)
    pos = pos.unsqueeze(0)
    targets = torch.cat([pos, negs], dim=0)

    logits = torch.cosine_similarity(proj_x.float(), targets.float(), dim=-1).type_as(proj_x)
    logits /= logit_temp
    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")
    logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
    return logits


class LogitGenerator(Module):
    """Generate the logits of masked and unmasked inputs.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        num_classes (int): The number of classes in the labels.
        final_dim (int): Project final representations and targets to `final_dim`.
        skip_masked (bool): If True, skip computing losses over masked frames.
        skip_nomask (bool): If True, skip computing losses over unmasked frames.
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        num_classes: int,
        final_dim: int,
        skip_masked: bool,
        skip_nomask: bool,
    ):
        super().__init__()
        self.label_embeddings = Parameter(torch.FloatTensor(num_classes, final_dim))
        torch.nn.init.uniform_(self.label_embeddings)
        self.final_proj = torch.nn.Linear(encoder_embed_dim, final_dim)
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask

    def forward(self, x: Tensor, label: Tensor, mask_m: Tensor, mask_u: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (Tensor): The feature representation of the last transformer layer.
            label (Tensor): The label Tensor of dimension `[batch, frame]`.
            mask_m (Tensor): The masked indices of dimension `[batch, frame]`.
            mask_u (Tensor): The unmasked indices of dimension `[batch, frame]`.

        Returns:
            Tensor: The logits of masked frames. Tensor of dimension `[masked_frame, final_dim]`.
            Tensor: The logits of unmasked frames. Tensor of dimension `[unmasked_frame, final_dim]`.
        """
        proj_x = self.final_proj(x)
        if self.skip_masked:
            logit_m = None
        else:
            proj_x_m = proj_x[mask_m]
            label_m = label[mask_m]
            logit_m = _compute_logits(proj_x_m, label_m, self.label_embeddings)

        if self.skip_nomask:
            logit_u = None
        else:
            proj_x_u = proj_x[mask_u]
            label_u = label[mask_u]
            logit_u = _compute_logits(proj_x_u, label_u, self.label_embeddings)
        return logit_m, logit_u


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
