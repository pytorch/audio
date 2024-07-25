from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torchaudio.models import Wav2Vec2Model
from torchaudio.models.conformer import ConformerLayer
from torchaudio.models.rnnt import _TimeReduction
from torchaudio.models.wav2vec2 import components


def _buffered_arange(max) -> Tensor:
    """Compute arange using a buffered tensor across function calls.
    Produces same result as torch.arange(end=max).

    Args:
        max (int): Ending value for arange.
    """
    if not hasattr(_buffered_arange, "buf"):
        _buffered_arange.buf = torch.LongTensor()
    if max > _buffered_arange.buf.numel():
        _buffered_arange.buf.resize_(max)
        torch.arange(max, out=_buffered_arange.buf)
    return _buffered_arange.buf[:max]


def _sample_negatives(input: Tensor, num_negatives: int, cross_sample_negatives: int) -> Tuple[Tensor, Tensor]:
    """Sample negative examples from masked input.

    Args:
        input (Tensor): Tensor of dimension `(batch, frame, dim)`.
        num_negatives (int): Number of negative examples to sample.
        cross_sample_negatives (int): Number of negative examples to cross sample.

    Returns:
        (Tensor, Tensor):
        Tensor
            The negative samples.
        Tensor
            The indices of the negative samples.
    """
    if num_negatives == 0 and cross_sample_negatives == 0:
        return (
            torch.zeros(0).to(input.device, input.dtype),
            torch.zeros(0).to(input.device, input.dtype),
        )

    B, T, D = input.shape
    input = input.view(-1, D)

    cross_high = T * B
    high = T

    assert high > 1

    if num_negatives > 0:
        tszs = _buffered_arange(T).unsqueeze(-1).expand(-1, num_negatives).flatten()

        neg_idxs = torch.randint(low=0, high=high - 1, size=(B, num_negatives * T))
        neg_idxs[neg_idxs >= tszs] += 1

    if cross_sample_negatives > 0:
        tszs = _buffered_arange(T).unsqueeze(-1).expand(-1, cross_sample_negatives).flatten()

        cross_neg_idxs = torch.randint(low=0, high=cross_high - 1, size=(B, cross_sample_negatives * T))
        cross_neg_idxs[cross_neg_idxs >= tszs] += 1

    if num_negatives > 0:
        neg_idxs = neg_idxs + (torch.arange(B).unsqueeze(1) * high)
    else:
        neg_idxs = cross_neg_idxs

    if cross_sample_negatives > 0 and num_negatives > 0:
        neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

    negs = input[neg_idxs.view(-1)]
    negs = negs.view(B, T, num_negatives + cross_sample_negatives, D).permute(2, 0, 1, 3)  # NxBxCxT

    return negs, neg_idxs


class NegativeSampler(Module):
    r"""Applies preprocessing to input and then computes negative sampling.

    Args:
        preprocessor (nn.Module): Transforms input tensor prior to negative sampling.
        num_negatives (int): Number of negative examples to sample.
        cross_sample_negatives (int): Number of negative examples to cross sample.
    """

    def __init__(
        self,
        preprocessor: Module,
        num_negatives: int,
        cross_sample_negatives: int,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.num_negatives = num_negatives
        self.cross_sample_negatives = cross_sample_negatives

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            input (Tensor): Tensor of dimension `(B, T, D)`.

        Returns:
            (Tensor, Tensor, Optional[Tensor]):
            Tensor
                The input tensor after preprocessing, prior to being sampled.
            Tensor
                The negative samples.
            Tensor
                The indices of the negative samples.
        """
        preprocessed = self.preprocessor(input)
        negs, neg_idxs = _sample_negatives(preprocessed, self.num_negatives, self.cross_sample_negatives)
        return preprocessed, negs, neg_idxs


class FeatureEncoder(Module):
    """Feature Encoder class, consisting of time reduction and linear layer.

    Args:
        stride (int): Number of frames to merge for the output frame.
        input_dim (int): Input dimension of the tensor.
        output_dim (int): Output dimension of the tensor.
    """

    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.time_reduction_layer = _TimeReduction(stride=stride)
        self.linear_layer = nn.Linear(input_dim * stride, output_dim)

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x (Tensor): Feature Tensor representing log Mel Spectrogram output. shape ``(B, T, D)``.
            lengths (Tensor or None):
                Valid length of each input sample. shape: ``(B, )``.

        Returns:
            (Tensor, Optional[Tensor]):
                Tensor: output sequence after undergoing time reduction and linear projection.
                    Shape ``(B, T // stride, D * stride).
                Optional[Tensor]: output lengths of shape ``(B,)`` if lengths parameter is provided,
                    otherwise `None`.
        """
        if lengths is None:
            B, T, D = x.shape
            dummy_lengths = torch.full((B,), T)
            x, _ = self.time_reduction_layer(x, dummy_lengths)
            x = self.linear_layer(x)
            return x, None

        x, lengths = self.time_reduction_layer(x, lengths)
        x = self.linear_layer(x)
        return x, lengths


class ConformerEncoder(Module):
    """Conformer Encoder class, consisting of feature projection and conformer modules.

    Args:
        feature_projection (nn.Module):
            Projects feature to encoder dimension.
        conformer (nn.ModuleList)
            List of Conformer layers.
    """

    def __init__(
        self,
        feature_projection: Module,
        conformer: ModuleList,
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

    def _get_intermediate_outputs(
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

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            features (Tensor): Tensor of features of shape ``(B, T, D)``.
            lengths (Tensor or None, optional): Valid length of each input sample. shape: ``(B, )``.

        Returns:
            Tensor: result after applying conformer encoder to features.
        """
        x, mask = self._preprocess(features, lengths)
        x = x.transpose(0, 1)
        for layer in self.conformer:
            x = layer(x, mask)
        return x.transpose(0, 1)

    def extract_features(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> List[Tensor]:
        """Returns the list of outputs from the intermediate layers of conformer block in the encoder.

        Args:
            features (Tensor): Tensor of features of shape ``(B, T, D)``.
            lengths (Tensor or None, optional): Valid length of each input sample. shape: ``(B, )``.

        Returns:
            List[Tensor]:
                Features from requested layers. Each Tensor is of shape: `(batch, time frame, feature dimension)`.
        """
        x, masks = self._preprocess(features, lengths)
        return self._get_intermediate_outputs(x, mask=masks, num_layers=num_layers)


class ConformerWav2Vec2PretrainModel(Module):
    """Conformer Wav2Vec2 pre-train model for training from scratch.

    Note:
        To build the model, please use one of the factory functions,
        :py:func:`conformer_wav2vec2_base` or :py:func:`conformer_wav2vec2_large`

    Args:
        wav2vec2 (nn.Module):
            Conformer based Wav2Vec2 model, including feature extractor and conformer encoder components.
        mask_generator (nn.Module):
            Mask generator that generates the mask for masked prediction during training.
        negative_sampler (nn.Module):
            Negative sampler to apply after masking.

    """

    def __init__(
        self,
        wav2vec2: Wav2Vec2Model,
        mask_generator: Module,
        negative_sampler: Module,
    ):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.mask_generator = mask_generator
        self.negative_sampler = negative_sampler

    def forward(
        self,
        features: Tensor,
        audio_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Args:
            features (Tensor):
                Tensor of audio features of shape `(batch, frame, dim)`.
            audio_lengths (Tensor or None, optional):
                Tensor of valid length of each valid auidio in the batch.
                shape: `(batch, )` (Default: ``None``)

        Returns:
            (Tensor, Optional[Tensor], Tensor, Tensor, Tensor, Tensor):
            Tensor
                The masked sequences of probability distribution of shape `(batch, frame dim)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )` representing
                valid length in time axis is returns.
            Tensor
                The mask indices.
            Tensor
                The targets, prior to negative sampling.
            Tensor
                The negative samples.
            Tensor
                The indices of the negative samples.
        """
        x, lengths = self.wav2vec2.feature_extractor(features, audio_lengths)

        if lengths is not None:
            padding_mask = components._get_padding_mask(x, lengths)
        else:
            padding_mask = None

        x = self.wav2vec2.encoder.feature_projection.layer_norm(x)
        x = self.wav2vec2.encoder.feature_projection.dropout(x)

        # Unmasked feature is used to generate positive and negative samples.
        unmasked_x = x.clone()
        # Apply masking to x before passing it to Conformer layers.
        x, mask_idxs = self.mask_generator(x, padding_mask)
        # Select the frames from masked indices for negative sampling.
        unmasked_x = unmasked_x[mask_idxs].view(x.shape[0], -1, x.shape[-1])
        targets, negs, neg_idxs = self.negative_sampler(unmasked_x)

        x = self.wav2vec2.encoder.feature_projection.projection(x)
        x = x.transpose(0, 1)
        for conformer_layer in self.wav2vec2.encoder.conformer:
            x = conformer_layer(x, padding_mask)
        x = x.transpose(0, 1)

        return x, lengths, mask_idxs, targets, negs, neg_idxs


################################################################################
def _get_conformer_feature_extractor(
    input_dim: int,
    output_dim: int,
    stride: int,
) -> FeatureEncoder:
    """Construct Feature Extractor

    Args:
        input_dim (int): Input dimension of features.
        output_dim (int): Output dimension after feature extraction.
        stride (int): Stride used in Time Reduction layer of feature extractor.

    Returns:
        FeatureEncoder: The resulting feature extraction.
    """
    return FeatureEncoder(input_dim, output_dim, stride)


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
        in_features (int): The number of input features.
        embed_dim (int): The dimension of the embedding in the feature projection.
        dropout_input (float): The dropout probability applied after the input feature
            is projected to ``embed_dim``.
        num_layers (int): Number of Conformer layers in the encoder.
        num_heads (int): Number of heads in each Conformer layer.
        ff_interm_features (int): Hidden layer dimension of the feedforward network in
            each Conformer layer.
        dropout (float): Dropout probability in each Conformer layer.
        depthwise_conv_kernel_size (int or List[int]): List of kernel sizes corresponding
            to each of the  Conformer layers.If int is provided, all layers will have the
            same kernel size.
        convolution_first (bool): Whether to apply the convolution module ahead of the
            attention module in each Conformer layer.
        use_group_norm (bool): Whether to use ``GroupNorm`` rather than ``BatchNorm1d`` in
            the convolution module in each Conformer layer.

    Returns:
        ConformerEncoder:
            The resulting conformer encoder module.
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

    return ConformerEncoder(feature_projection, ModuleList(conformer_layers))


def _get_conformer_negativer_sampler(
    input_dim: int,
    output_dim: int,
    num_negatives: int,
    cross_sample_negatives: int,
) -> NegativeSampler:
    """Build custom NegativeSampler module, including linear layer and negative sampling.

    Args:
        input_dim (int): Dimension of input after feature extraction.
        output_dim (int): Dimension of embedding for use in negative sampling. Same as the
            embedding in the feature projection.
        num_negatives (int): Number of negatives to sample.
        cross_sample_negatives (int): Number of cross sampled negatives.

    Returns:
        NegativeSampler:
            The resulting negative sampler module.
    """
    preprocessor = nn.Linear(input_dim, output_dim)
    return NegativeSampler(preprocessor, num_negatives, cross_sample_negatives)


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
) -> Wav2Vec2Model:
    """Build a custom Conformer Wav2Vec2Model

    Args:
        extractor_input_dim (int): Input dimension of the features.
        extractor_output_dim (int): Output dimension after feature extraction.
        extractor_stride (int): Stride used in time reduction layer of feature extraction.
        encoder_embed_dim (int): The dimension of the embedding in the feature projection.
        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected to ``embed_dim``
        encoder_num_layers (int): Number of Conformer layers in the encoder.
        encoder_num_heads (int): Number of heads in each Conformer layer.
        encoder_ff_interm_features (int):
            Hidden layer dimension of the feedforward network in each Conformer layer.
        encoder_depthwise_conv_kernel_size (int or List[int]):
            List of kernel sizes corresponding to each of the Conformer layers.
            If int is provided, all layers will have the same kernel size.
        encoder_dropout (float): Dropout probability in each Conformer layer.
        encoder_convolution_first (bool):
            Whether to apply the convolution module ahead of the attention module
            in each Conformer layer.
        encoder_use_group_norm (bool):
            Whether to use ``GroupNorm`` rather than ``BatchNorm1d`` in the convolution
            module in each Conformer layer.

    Returns:
        Wav2Vec2Model:
            The resulting wav2vec2 model with a conformer encoder.
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

    return Wav2Vec2Model(feature_extractor, encoder)


def conformer_wav2vec2_base(
    extractor_input_dim: int = 64,
    extractor_output_dim: int = 256,
    encoder_projection_dropout: float = 0.0,
) -> Wav2Vec2Model:
    """
    Build Conformer Wav2Vec2 Model with "small" architecture from
    *Conformer-Based Slef-Supervised Learning for Non-Speech Audio Tasks* :cite:`9746490`

    Args:
        extractor_input_dim (int, optional): Input dimension of feature extractor. (Default: 64)
        extractor_output_dim (int, optional): Output dimension of feature extractor. (Default: 256)
        encoder_projection_dropout (float, optional):
            Dropout probability applied after feature projection. (Default: 0.0)

    Returns:
        Wav2Vec2Model:
            The resulting wav2vec2 model with a conformer encoder and ``base`` configuration.
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


def conformer_wav2vec2_pretrain_model(
    extractor_input_dim: int,
    extractor_output_dim: int,
    extractor_stride: int,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_ff_interm_features: int,
    encoder_depthwise_conv_kernel_size: int,
    encoder_dropout: float,
    encoder_convolution_first: bool,
    encoder_use_group_norm: bool,
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
    num_negatives: int,
    cross_sample_negatives: int,
) -> ConformerWav2Vec2PretrainModel:
    """Build a custom Conformer Wav2Vec2 Model for pre-training

    Args:
        extractor_input_dim (int): Input dimension of the features.
        extractor_output_dim (int): Output dimension after feature extraction.
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
        mask_prob (float):
            Probability for each token to be chosen as start of the span to be masked.
        mask_selection (str)
            How to choose the mask length. Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_other (float):
            Secondary mask argument (used for more complex distributions).
        mask_length (int):
            The lengths of the mask.
        no_mask_overlap (bool):
            Whether to allow masks to overlap.
        mask_min_space (int):
            Minimum space between spans (if no overlap is enabled).
        mask_channel_prob: (float):
            The probability of replacing a feature with 0.
        mask_channel_selection (str):
            How to choose the mask length for channel masking.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_channel_other (float):
            Secondary mask argument for channel masking (used for more complex distributions).
        mask_channel_length (int):
            Minimum space between spans (if no overlap is enabled) for channel masking.
        no_mask_channel_overlap (bool):
            Whether to allow channel masks to overlap.
        mask_channel_min_space (int):
            Minimum space between spans for channel masking (if no overlap is enabled).
        num_negatives (int):
            Number of negatives to sample.
        cross_sample_negatives (int):
            Number of cross sampled negatives.

    Returns:
        ConformerWav2Vec2PretrainModel:
            The resulting model.
    """
    wav2vec2 = conformer_wav2vec2_model(
        extractor_input_dim,
        extractor_output_dim,
        extractor_stride,
        encoder_embed_dim,
        encoder_projection_dropout,
        encoder_num_layers,
        encoder_num_heads,
        encoder_ff_interm_features,
        encoder_depthwise_conv_kernel_size,
        encoder_dropout,
        encoder_convolution_first,
        encoder_use_group_norm,
    )

    mask_generator = components.MaskGenerator(
        extractor_output_dim,
        mask_prob,
        mask_selection,
        mask_other,
        mask_length,
        no_mask_overlap,
        mask_min_space,
        mask_channel_prob,
        mask_channel_selection,
        mask_channel_other,
        mask_channel_length,
        no_mask_channel_overlap,
        mask_channel_min_space,
    )

    negative_sampler = _get_conformer_negativer_sampler(
        extractor_output_dim,
        encoder_embed_dim,
        num_negatives,
        cross_sample_negatives,
    )

    return ConformerWav2Vec2PretrainModel(
        wav2vec2=wav2vec2,
        mask_generator=mask_generator,
        negative_sampler=negative_sampler,
    )


def conformer_wav2vec2_pretrain_base(
    extractor_input_dim: int = 64,
    extractor_output_dim: int = 256,
    encoder_projection_dropout: float = 0.0,
    mask_prob: float = 0.3,
    mask_length: int = 3,
    num_negatives: int = 100,
    cross_sample_negatives: int = 0,
) -> ConformerWav2Vec2PretrainModel:
    """Build Conformer Wav2Vec2 Model for pre-training with "small" architecture from
    *Conformer-Based Self-Supervised Learning for Non-Speech Audio Tasks* :cite:`9746490`

    Args:
        extractor_input_dim (int, optional): Input dimension of the features. (Default: 64)
        extractor_output_dim (int, optional): Output dimension after feature extraction. (Default: 256)
        encoder_projection_dropout (float, optional):
            The dropout probability applied after the input feature is projected to
            ``embed_dim``. (Default: 0.0)
        mask_prob (float, optional):
            Probability for each token to be chosen as start of the span to be masked. (Default: 0.3)
        mask_length (int, optional):
            The lengths of the mask. (Default: 3)
        num_negatives (int, optional):
            Number of sampled negatives. (Default: 0)
        cross_sample_negatives (int, optional):
            Number of cross sampled negatives. (Default: 0)

    Returns:
        ConformerWav2Vec2PretrainModel:
            The resulting model.
    """
    return conformer_wav2vec2_pretrain_model(
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
        mask_prob=mask_prob,
        mask_selection="static",
        mask_other=0.0,
        mask_length=mask_length,
        no_mask_overlap=False,
        mask_min_space=0,
        mask_channel_prob=0,
        mask_channel_selection="static",
        mask_channel_other=0,
        mask_channel_length=10,
        no_mask_channel_overlap=False,
        mask_channel_min_space=1,
        num_negatives=num_negatives,
        cross_sample_negatives=cross_sample_negatives,
    )


def conformer_wav2vec2_pretrain_large(
    extractor_input_dim: int = 64,
    extractor_output_dim: int = 256,
    encoder_projection_dropout: float = 0.0,
    mask_prob: float = 0.3,
    mask_length: int = 3,
    num_negatives: int = 100,
    cross_sample_negatives: int = 0,
) -> ConformerWav2Vec2PretrainModel:
    """Build Conformer Wav2Vec2 Model for pre-training with "large" architecture from
    *Conformer-Based Slef-Supervised Learning for Non-Speech Audio Tasks* :cite:`9746490`

    Args:
        extractor_input_dim (int, optional): Input dimension of the features. (Default: 64)
        extractor_output_dim (int, optional): Output dimension after feature extraction. (Default: 256)
        encoder_projection_dropout (float, optional):
            The dropout probability applied after the input feature is projected to
            ``embed_dim``. (Default: 0.0)
        mask_prob (float, optional):
            Probability for each token to be chosen as start of the span to be masked. (Default: 0.3)
        mask_length (int, optional):
            The lengths of the mask. (Default: 3)
        num_negatives (int, optional):
            Number of sampled negatives. (Default: 0)
        cross_sample_negatives (int, optional):
            Number of cross sampled negatives. (Default: 0)

    Returns:
        ConformerWav2Vec2PretrainModel:
            The resulting model.
    """
    return conformer_wav2vec2_pretrain_model(
        extractor_input_dim=extractor_input_dim,
        extractor_output_dim=extractor_output_dim,
        extractor_stride=4,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_ff_interm_features=1024,
        encoder_depthwise_conv_kernel_size=[31] + [15] * 11,
        encoder_dropout=0.1,
        encoder_convolution_first=True,
        encoder_use_group_norm=True,
        mask_prob=mask_prob,
        mask_selection="static",
        mask_other=0.0,
        mask_length=mask_length,
        no_mask_overlap=False,
        mask_min_space=0,
        mask_channel_prob=0,
        mask_channel_selection="static",
        mask_channel_other=0,
        mask_channel_length=10,
        no_mask_channel_overlap=False,
        mask_channel_min_space=1,
        num_negatives=num_negatives,
        cross_sample_negatives=cross_sample_negatives,
    )
