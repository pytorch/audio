from typing import List, Optional, Tuple, Union

from torch import nn, Tensor
from torch.nn import Module
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components, model


class TimeReduction(Module):
    """Extract features from input. Consists of time reduction and linear layer.

    Args:
        stride (int): number of frames to merge for the output frame
        input_dim (int): input dimension of the tensor
        output_dim (int): output dimension of the tensor
    """

    def __init__(self, stride: int, input_dim: int, output_dim: int):
        super().__init__()
        self.time_reduction_layer = _TimeReduction(stride=stride)
        self.linear_layer = nn.Linear(input_dim * stride, output_dim)

    def forward(
        self,
        x: Tensor,
        length: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor):
                Input Tensor representing Mel Spectrogram output. shape ``[batch, frame, dim]``.
            length (Tensor or None, optional):
                Valid length of each input sample. shape: ``[batch, ]``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                Result of input tensor after passing through time reduction and linear layer
            Tensor or None
                If ``length`` is provided, a Tensor of shape `(batch, )` is returned, indicating
                the valid length of each feature in the Tensor.
        """
        # TODO: case if length is none?
        if length is not None:
            x, length = self.time_reduction_layer(x, length)
        x = self.linear_layer(x)
        return x, length


class ConformerEncoder(Module):
    """Conformer Encoder class, consisting of feature projection and conformer modules.

    Args:
        feature_projection (nn.Module):
            Projects feature to encoder dimension
        conformer (nn.ModuleList)
            List of Conformer layers
    """

    def __init__(
        self,
        feature_projection: Module,
        conformer: nn.ModuleList,
    ):
        super().__init__()
        self.feature_projection = feature_projection
        self.conformer = conformer

    def _preprocess(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.feature_projection(features)
        if lengths is not None:
            mask = components._get_padding_mask(x, lengths)
        else:
            mask = None
        return x, mask

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        x, mask = self._preprocess(features, lengths)
        x = x.transpose(0, 1)
        for layer in self.conformer:
            x = layer(x, mask)
        return x.transpose(0, 1)

    def get_intermediate_outputs(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        if num_layers is not None:
            if not 0 < num_layers <= len(self.conformer):
                raise ValueError(f"`num_layers` must be between [1, {len(self.conformer)}]")

        ret: List[Tensor] = []

        x = x.transpose(0, 1)
        for layer in self.conformer:
            x = layer(x, mask)
            ret.append(x.transpose(0, 1))
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        x, masks = self._preprocess(features, lengths)
        return self.get_intermediate_outputs(x, mask=masks, num_layers=num_layers)


################################################################################
def _get_conformer_feature_extractor(
    input_dim: int,
    output_dim: int,
    stride: int,
) -> TimeReduction:
    """Construct Feature Extractor

    Args:
        input_dim (int): Input dimension of features
        output_dim (int): Output dimension after feature extraction
        stride (int): Stride used in Time Reduction layer of feature extractor
    """
    return TimeReduction(stride, input_dim, output_dim)


def _get_conformer_encoder(
    in_features: int,
    embed_dim: int,
    dropout_input: float,
    num_layers: int,
    num_heads: int,
    ff_interm_features: int,
    dropout: float,
    depthwise_conv_kernel_size: Union[int, List[int]],
    convolution_first: bool,
    use_group_norm: bool,
) -> ConformerEncoder:
    """Construct Conformer Encoder

    Args:
        in_features (int):
            The number of input features.
        embed_dim (int):
            The dimension of the embedding in the feature projection.
        dropout_input (float):
            The dropout probability applied after the input feature is projected to
            ``embed_dim``.
        num_layers (int):
            Number of Conformer layers in the encoder.
        num_heads (int):
            Number of heads in each Conformer layer.
        ff_interm_features (int):
            Hidden layer dimension of the feedforward network in each Conformer layer.
        dropout (float):
            Dropout probability in each Conformer layer.
        depthwise_conv_kernel_size (int or List[int]):
            List of kernel sizes corresponding to each of the Conformer layers.
            If int is provided, all layers will have the same kernel size.
        convolution_first (bool):
            Whether to apply the convolution module ahead of the attention module
            in each Conformer layer.
        use_group_norm (bool):
            Whether to use ``GroupNorm`` rather than ``BatchNorm1d`` in the convolution
            module in each Conformer layer.
    """
    feature_projection = components.FeatureProjection(in_features, embed_dim, dropout_input)

    if type(depthwise_conv_kernel_size) == int:
        depthwise_conv_kernel_size = [depthwise_conv_kernel_size] * num_layers

    assert len(depthwise_conv_kernel_size) == num_layers

    conformer_layers = []
    for l in range(num_layers):
        layer = ConformerLayer(
            input_dim=embed_dim,
            ffn_dim=ff_interm_features,
            num_attention_heads=num_heads,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size[l],
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )
        conformer_layers.append(layer)

    return ConformerEncoder(feature_projection, nn.ModuleList(conformer_layers))


def conformer_wav2vec2_model(
    extractor_input_dim: int,
    extractor_output_dim: int,
    extractor_stride: int,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_ff_interm_features: int,
    encoder_depthwise_conv_kernel_size: Union[int, List[int]],
    encoder_dropout: float,
    encoder_convolution_first: bool,
    encoder_use_group_norm: bool,
):
    """Build a custom Conformer Wav2Vec2Model

    Args:
        extractor_input_dim (int):
            Input dimension of the features.
        extractor_output_dim (int):
            Output dimension after feature extraction.
        extractor_stride (int):
            Stride used in time reduction layer of feature extraction.
        encoder_embed_dim (int):
            The dimension of the embedding in the feature projection.
        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected to
            ``embed_dim``
        encoder_num_layers (int):
            Number of Conformer layers in the encoder.
        encoder_num_heads (int):
            Number of heads in each Conformer layer.
        encoder_ff_interm_features (int):
            Hidden layer dimension of the feedforward network in each Conformer layer.
        encoder_depthwise_conv_kernel_size (int or List[int]):
            List of kernel sizes corresponding to each of the Conformer layers.
            If int is provided, all layers will have the same kernel size.
        encoder_dropout (float):
            Dropout probability in each Conformer layer.
        encoder_convolution_first (bool):
            Whether to apply the convolution module ahead of the attention module
            in each Conformer layer.
        encoder_use_group_norm (bool):
            Whether to use ``GroupNorm`` rather than ``BatchNorm1d`` in the convolution
            module in each Conformer layer.
    """
    feature_extractor = _get_conformer_feature_extractor(
        extractor_input_dim,
        extractor_output_dim,
        extractor_stride,
    )

    encoder = _get_conformer_encoder(
        in_features=extractor_output_dim,
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        ff_interm_features=encoder_ff_interm_features,
        depthwise_conv_kernel_size=encoder_depthwise_conv_kernel_size,
        dropout=encoder_dropout,
        convolution_first=encoder_convolution_first,
        use_group_norm=encoder_use_group_norm,
    )

    return model.Wav2Vec2Model(feature_extractor, encoder)


def conformer_wav2vec2_base(
    extractor_input_dim: int = 64,
    extractor_output_dim: int = 256,
    encoder_projection_dropout: float = 0.0,
):
    """
    Build Conformer Wav2Vec2 Model with "small" architecture from
    *Conformer-Based Slef-Supervised Learning for Non-Speech Audio Tasks* :cite:`9746490`

    Args:
        extractor_input_dim (int, optional):
            Input dimension of feature extractor. (Default: 64)
        extractor_output_dim (int, optional):
            Output dimension of feature extractor. (Default: 256)
        encoder_projection_dropout (float, optional):
            Dropout probability applied after feature projection. (Default: 0.0)
    """
    return conformer_wav2vec2_model(
        extractor_input_dim=extractor_input_dim,
        extractor_output_dim=extractor_output_dim,
        extractor_stride=4,
        encoder_embed_dim=256,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_num_layers=12,
        encoder_num_heads=8,
        encoder_ff_interm_features=1024,
        encoder_depthwise_conv_kernel_size=[31] + [15] * 11,
        encoder_dropout=0.1,
        encoder_convolution_first=True,
        encoder_use_group_norm=True,
    )
