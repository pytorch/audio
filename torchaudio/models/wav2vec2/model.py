from typing import Optional, Tuple, List

import torch
from torch import Tensor
from torch.nn import Module

from . import components


class Wav2Vec2Model(Module):
    """torchaudio.models.Wav2Vec2Model(feature_extractor: torch.nn.Module, encoder: torch.nn.Module, aux: Optional[torch.nn.Module] = None)

    Encoder model used in *wav2vec 2.0* [:footcite:`baevski2020wav2vec`].

    Note:
        To build the model, please use one of the factory functions.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.

        aux (torch.nn.Module or None, optional):
            Auxiliary module. If provided, the output from encoder is passed to this module.
    """  # noqa: E501

    def __init__(
        self,
        feature_extractor: Module,
        encoder: Module,
        aux: Optional[Module] = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.aux = aux

    @torch.jit.export
    def extract_features(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
        num_layers: Optional[int] = None,
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """Extract feature vectors from raw waveforms

        This returns the list of outputs from the intermediate layers of
        transformer block in encoder.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that the entire audio waveform
                length is valid.
            num_layers (int or None, optional):
                If given, limit the number of intermediate layers to go through.
                Providing `1` will stop the computation after going through one
                intermediate layers. If not given, the outputs from all the
                intermediate layers are returned.

        Returns:
            (List[Tensor], Optional[Tensor]):
            List of Tensors
                Features from requested layers.
                Each Tensor is of shape: `(batch, time frame, feature dimension)`
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of each feature Tensor.
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder.extract_features(x, lengths, num_layers)
        return x, lengths

    def forward(
        self,
        waveforms: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape `(batch, frames)`.
            lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `(batch, )`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Optional[Tensor]):
            Tensor
                The sequences of probability distribution (in logit) over labels.
                Shape: `(batch, frames, num labels)`.
            Tensor or None
                If ``lengths`` argument was provided, a Tensor of shape `(batch, )`
                is returned.
                It indicates the valid length in time axis of the output Tensor.
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        x = self.encoder(x, lengths)
        if self.aux is not None:
            x = self.aux(x)
        return x, lengths


class HuBERTPretrainModel(Module):
    """HuBERT pre-train model for training from scratch.

    Note:
        To build the model, please use one of the factory functions in
            `[hubert_pretrain_base, hubert_pretrain_large, hubert_pretrain_xlarge]`.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.

        mask_generator (torch.nn.Module):
            Mask generator that generates the mask for masked prediction during the training.

        logit_generator (torch.nn.Module):
            Logit generator that predicts the logits of the masked and unmasked inputs.
    """

    def __init__(
        self,
        wav2vec2: Wav2Vec2Model,
        mask_generator: Module,
        logit_generator: Module,
    ):
        super().__init__()
        self.wav2vec2 = wav2vec2
        self.mask_generator = mask_generator
        self.logit_generator = logit_generator

    def forward(
        self,
        waveforms: Tensor,
        labels: Tensor,
        audio_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of dimension `[batch, frames]`.
            labels (Tensor): Label for pre-training. A Tensor of dimension `[batch, frames]`.
            audio_lengths (Tensor or None, optional):
                Indicates the valid length of each audio in the batch.
                Shape: `[batch, ]`.
                When the ``waveforms`` contains audios with different durations,
                by providing ``lengths`` argument, the model will compute
                the corresponding valid output lengths and apply proper mask in
                transformer attention layer.
                If ``None``, it is assumed that all the audio in ``waveforms``
                have valid length. Default: ``None``.

        Returns:
            (Tensor, Tensor, Tensor):
            Tensor
                The masked sequences of probability distribution (in logit).
                Shape: `(masked_frames, num labels)`.
            Tensor
                The unmasked sequence of probability distribution (in logit).
                Shape: `(unmasked_frames, num labels)`.
            Tensor
                The feature mean value for additional penalty loss.
                Shape: `(1,)`.
        """
        x, lengths = self.wav2vec2.feature_extractor(waveforms, audio_lengths)
        features_pen = x.float().pow(2).mean()
        if lengths is not None:
            padding_mask = components._get_padding_mask(x, lengths)
        else:
            padding_mask = None
        x, attention_mask = self.wav2vec2.encoder._preprocess(x, lengths)
        x, mask = self.mask_generator(x, padding_mask)
        x = self.wav2vec2.encoder.transformer(x, attention_mask=attention_mask)
        assert x.shape[1] == labels.shape[1], "The length of label must match that of HuBERT model output"
        if padding_mask is not None:
            mask_m = torch.logical_and(~padding_mask, mask)
            mask_u = torch.logical_and(~padding_mask, ~mask_m)
        else:
            mask_m = mask
            mask_u = ~mask_m

        logit_m, logit_u = self.logit_generator(x, labels, mask_m, mask_u)

        return logit_m, logit_u, features_pen


def wav2vec2_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
    aux_num_out: Optional[int],
) -> Wav2Vec2Model:
    # Overriding the signature so that the return type is correct on Sphinx
    """wav2vec2_model(extractor_mode: str, extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]], extractor_conv_bias: bool, encoder_embed_dim: int, encoder_projection_dropout: float, encoder_pos_conv_kernel: int, encoder_pos_conv_groups: int, encoder_num_layers: int, encoder_num_heads: int, encoder_attention_dropout: float, encoder_ff_interm_features: int, encoder_ff_interm_dropout: float, encoder_dropout: float, encoder_layer_norm_first: bool, encoder_layer_drop: float, aux_num_out: Optional[int]) -> torchaudio.models.Wav2Vec2Model

    Build a custom Wav2Vec2Model

    Note:
        The "feature extractor" below corresponds to
        `ConvFeatureExtractionModel <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736>`__
        in the original ``fairseq`` implementation.
        This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
        [:footcite:`baevski2020wav2vec`] paper.

        The "encoder" below corresponds to `TransformerEncoder <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817>`__,
        and this is referred as "Transformer" in the paper.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            Valid values are ``"group_norm"`` or ``"layer_norm"``.
            If ``"group_norm"``, then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.

            This option corresponds to ``extractor_mode`` from ``fairseq``.
        extractor_conv_layer_config (list of integer tuples or None):
            Configuration of convolution layers in feature extractor.
            List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``

            If ``None`` is provided, then the following default value is used.

            .. code-block:: python

               [
                 (512, 10, 5),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 2, 2),
                 (512, 2, 2),
               ]

            This option corresponds to ``conv_feature_layers`` from ``fairseq``.

        extractor_conv_bias (bool):
            Whether to include bias term to each convolution operation.

            This option corresponds to ``conv_bias`` from ``fairseq``.

        encoder_embed_dim (int):
            The dimension of embedding in encoder.

            This option corresponds to ``encoder_embed_dim`` from ``fairseq``.

        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected
            to ``encoder_embed_dim``.

            This option corresponds to ``dropout_input`` from ``fairseq``.

        encoder_pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.

            This option corresponds to ``conv_pos`` from ``fairseq``.

        encoder_pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.

            This option corresponds to ``conv_pos_groups`` from ``fairseq``.

        encoder_num_layers (int):
            The number of self attention layers in transformer block.

            This option corresponds to ``encoder_layers`` from ``fairseq``.

        encoder_num_heads (int):
            The number of heads in self attention layers.

            This option corresponds to ``encoder_attention_heads`` from ``fairseq``.

        encoder_attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.

            This option corresponds to ``attention_dropout`` from ``fairseq``.

        encoder_ff_interm_features (int):
            The dimension of hidden features in feed forward layer.

            This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.

        encoder_ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.

            This option correspinds to ``activation_dropout`` from ``fairseq``.

        encoder_dropout (float):
            The dropout probability applied at the end of feed forward layer.

            This option corresponds to ``dropout`` from ``fairseq``.

        encoder_layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.

            This option corresponds to ``layer_norm_first`` from ``fairseq``.

        encoder_layer_drop (float):
            Probability to drop each encoder layer during training.

            This option corresponds to ``layerdrop`` from ``fairseq``.

        aux_num_out (int or None):
            When provided, attach an extra linear layer on top of encoder, which can be
            used for fine-tuning.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias
    )
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    aux = None
    if aux_num_out is not None:
        aux = torch.nn.Linear(in_features=encoder_embed_dim, out_features=aux_num_out)
    return Wav2Vec2Model(feature_extractor, encoder, aux)


def wav2vec2_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    # Overriding the signature so that the return type is correct on Sphinx
    """wav2vec2_base(encoder_projection_dropout: float = 0.1, encoder_attention_dropout: float = 0.1, encoder_ff_interm_dropout: float = 0.1, encoder_dropout: float = 0.1, encoder_layer_drop: float = 0.1, aux_num_out: Optional[int] = None) -> torchaudio.models.Wav2Vec2Model

    Build Wav2Vec2Model with "base" architecture from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def wav2vec2_large(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    # Overriding the signature so that the return type is correct on Sphinx
    """wav2vec2_large(encoder_projection_dropout: float = 0.1, encoder_attention_dropout: float = 0.1, encoder_ff_interm_dropout: float = 0.1, encoder_dropout: float = 0.1, encoder_layer_drop: float = 0.1, aux_num_out: Optional[int] = None) -> torchaudio.models.Wav2Vec2Model

    Build Wav2Vec2Model with "large" architecture from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def wav2vec2_large_lv60k(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.1,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.1,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    # Overriding the signature so that the return type is correct on Sphinx
    """wav2vec2_large_lv60k( encoder_projection_dropout: float = 0.1, encoder_attention_dropout: float = 0.0, encoder_ff_interm_dropout: float = 0.1, encoder_dropout: float = 0.0, encoder_layer_drop: float = 0.1, aux_num_out: Optional[int] = None) -> torchaudio.models.Wav2Vec2Model

    Build Wav2Vec2Model with "large lv-60k" architecture from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def hubert_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.05,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    # Overriding the signature so that the return type is correct on Sphinx
    """hubert_base(encoder_projection_dropout: float = 0.1, encoder_attention_dropout: float = 0.1, encoder_ff_interm_dropout: float = 0.0, encoder_dropout: float = 0.1, encoder_layer_drop: float = 0.05, aux_num_out: Optional[int] = None) -> torchaudio.models.Wav2Vec2Model

    Build HuBERT model with "base" architecture from *HuBERT* [:footcite:`hsu2021hubert`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def hubert_large(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    # Overriding the signature so that the return type is correct on Sphinx
    """hubert_large(encoder_projection_dropout: float = 0.0, encoder_attention_dropout: float = 0.0, encoder_ff_interm_dropout: float = 0.0, encoder_dropout: float = 0.0, encoder_layer_drop: float = 0.0, aux_num_out: Optional[int] = None) -> torchaudio.models.Wav2Vec2Model

    Build HuBERT model with "large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def hubert_xlarge(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
    aux_num_out: Optional[int] = None,
) -> Wav2Vec2Model:
    # Overriding the signature so that the return type is correct on Sphinx
    """hubert_xlarge(encoder_projection_dropout: float = 0.0, encoder_attention_dropout: float = 0.0, encoder_ff_interm_dropout: float = 0.0, encoder_dropout: float = 0.0, encoder_layer_drop: float = 0.0, aux_num_out: Optional[int] = None) -> torchaudio.models.Wav2Vec2Model

    Build HuBERT model with "extra large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_attention_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_dropout (float):
            See :py:func:`wav2vec2_model`.
        encoder_layer_drop (float):
            See :py:func:`wav2vec2_model`.
        aux_num_out (int or None, optional):
            See :py:func:`wav2vec2_model`.

    Returns:
        Wav2Vec2Model:
            The resulting model.
    """  # noqa: E501
    return wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1280,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=48,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=5120,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        aux_num_out=aux_num_out,
    )


def hubert_pretrain_model(
    extractor_mode: str,
    extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]],
    extractor_conv_bias: bool,
    encoder_embed_dim: int,
    encoder_projection_dropout: float,
    encoder_pos_conv_kernel: int,
    encoder_pos_conv_groups: int,
    encoder_num_layers: int,
    encoder_num_heads: int,
    encoder_attention_dropout: float,
    encoder_ff_interm_features: int,
    encoder_ff_interm_dropout: float,
    encoder_dropout: float,
    encoder_layer_norm_first: bool,
    encoder_layer_drop: float,
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
    skip_masked: bool,
    skip_nomask: bool,
    num_classes: int,
    final_dim: int,
) -> HuBERTPretrainModel:
    # Overriding the signature so that the return type is correct on Sphinx
    """hubert_pretrain_model(extractor_mode: str, extractor_conv_layer_config: Optional[List[Tuple[int, int, int]]], extractor_conv_bias: bool, encoder_embed_dim: int, encoder_projection_dropout: float, encoder_pos_conv_kernel: int, encoder_pos_conv_groups: int, encoder_num_layers: int, encoder_num_heads: int, encoder_attention_dropout: float, encoder_ff_interm_features: int, encoder_ff_interm_dropout: float, encoder_dropout: float, encoder_layer_norm_first: bool, encoder_layer_drop: float, mask_prob: float, mask_selection: str, mask_other: float, mask_length: int, no_mask_overlap: bool, mask_min_space: int, mask_channel_prob: float, mask_channel_selection: str, mask_channel_other: float, mask_channel_length: int, no_mask_channel_overlap: bool, mask_channel_min_space: int, skip_masked: bool, skip_nomask: bool, num_classes: int, final_dim: int) -> torchaudio.models.HuBERTPretrainModel

    Build a custom HuBERTPretrainModel for training from scratch

    Note:
        The "feature extractor" below corresponds to
        `ConvFeatureExtractionModel <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L736>`__
        in the original ``fairseq`` implementation.
        This is referred as "(convolutional) feature encoder" in the *wav2vec 2.0*
        [:footcite:`baevski2020wav2vec`] paper.

        The "encoder" below corresponds to `TransformerEncoder <https://github.com/pytorch/fairseq/blob/dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3/fairseq/models/wav2vec/wav2vec2.py#L817>`__,
        and this is referred as "Transformer" in the paper.

    Args:
        extractor_mode (str): Operation mode of feature extractor.
            Valid values are ``"group_norm"`` or ``"layer_norm"``.
            If ``"group_norm"``, then a single normalization is applied
            in the first convolution block. Otherwise, all the convolution
            blocks will have layer normalization.

            This option corresponds to ``extractor_mode`` from ``fairseq``.

        extractor_conv_layer_config (list of integer tuples or None):
            Configuration of convolution layers in feature extractor.
            List of convolution configuration,
            i.e. ``[(output_channel, kernel_size, stride), ...]``

            If ``None`` is provided, then the following default value is used.

            .. code-block:: python

               [
                 (512, 10, 5),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 3, 2),
                 (512, 2, 2),
                 (512, 2, 2),
               ]

            This option corresponds to ``conv_feature_layers`` from ``fairseq``.

        extractor_conv_bias (bool):
            Whether to include bias term to each convolution operation.

            This option corresponds to ``conv_bias`` from ``fairseq``.

        encoder_embed_dim (int):
            The dimension of embedding in encoder.

            This option corresponds to ``encoder_embed_dim`` from ``fairseq``.

        encoder_projection_dropout (float):
            The dropout probability applied after the input feature is projected
            to ``encoder_embed_dim``.

            This option corresponds to ``dropout_input`` from ``fairseq``.

        encoder_pos_conv_kernel (int):
            The kernel size of convolutional positional embeddings.

            This option corresponds to ``conv_pos`` from ``fairseq``.

        encoder_pos_conv_groups (int):
            The number of groups of convolutional positional embeddings.

            This option corresponds to ``conv_pos_groups`` from ``fairseq``.

        encoder_num_layers (int):
            The number of self attention layers in transformer block.

            This option corresponds to ``encoder_layers`` from ``fairseq``.

        encoder_num_heads (int):
            The number of heads in self attention layers.

            This option corresponds to ``encoder_attention_heads`` from ``fairseq``.

        encoder_attention_dropout (float):
            The dropout probability applied after softmax in self-attention layer.

            This option corresponds to ``attention_dropout`` from ``fairseq``.

        encoder_ff_interm_features (int):
            The dimension of hidden features in feed forward layer.

            This option corresponds to ``encoder_ffn_embed_dim`` from ``fairseq``.

        encoder_ff_interm_dropout (float):
            The dropout probability applied in feedforward layer.

            This option correspinds to ``activation_dropout`` from ``fairseq``.

        encoder_dropout (float):
            The dropout probability applied at the end of feed forward layer.

            This option corresponds to ``dropout`` from ``fairseq``.

        encoder_layer_norm_first (bool):
            Control the order of layer norm in transformer layer and each encoder layer.
            If True, in transformer layer, layer norm is applied before features are fed
            to encoder layers. In encoder layer, two layer norms are applied before and after
            self attention.
            If False, in transformer layer, layer norm is applied after features are fed
            to encoder layers. In encoder layer, two layer norms are applied after self
            attention, before and after feed forward.

            This option corresponds to ``layer_norm_first`` from ``fairseq``.

        encoder_layer_drop (float):
            Probability to drop each encoder layer during training.

            This option corresponds to ``layerdrop`` from ``fairseq``.

        mask_prob (float):
            Probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            However due to overlaps, the actual number will be smaller (unless no_overlap is True).

            This option corresponds to ``mask_prob`` from ``fairseq``.

        mask_selection (str):
            How to choose the mask length. Options: [``static``, ``uniform``, ``normal``, ``poisson``].

            This option corresponds to ``mask_selection`` from ``fairseq``.

        mask_other (float):
            Secondary mask argument (used for more complex distributions).

            This option corresponds to ``mask_other`` from ``fairseq``.

        mask_length (int):
            The lengths of the mask.

            This option corresponds to ``mask_length`` from ``fairseq``.

        no_mask_overlap (bool):
            Whether to allow masks to overlap.

            This option corresponds to ``no_mask_overlap`` from ``fairseq``.

        mask_min_space (int):
            Minimum space between spans (if no overlap is enabled).

            This option corresponds to ``mask_min_space`` from ``fairseq``.

        mask_channel_prob: (float):
            The probability of replacing a feature with 0.

            This option corresponds to ``mask_channel_prob`` from ``fairseq``.

        mask_channel_selection (str):
            How to choose the mask length for channel masking. Options: [``static``, ``uniform``, ``normal``, ``poisson``].

            This option corresponds to ``mask_channel_selection`` from ``fairseq``.

        mask_channel_other (float):
            Secondary mask argument for channel masking(used for more complex distributions).

            This option corresponds to ``mask_channel_other`` from ``fairseq``.

        mask_channel_length (int):
            Minimum space between spans (if no overlap is enabled) for channel masking.

            This option corresponds to ``mask_channel_length`` from ``fairseq``.

        no_mask_channel_overlap (bool):
            Whether to allow channel masks to overlap.

            This option corresponds to ``no_mask_channel_overlap`` from ``fairseq``.

        mask_channel_min_space (int):
            Minimum space between spans for channel masking(if no overlap is enabled).

            This option corresponds to ``mask_channel_min_space`` from ``fairseq``.

        skip_masked (bool):
            If True, skip computing losses over masked frames.

            This option corresponds to ``skip_masked`` from ``fairseq``.

        skip_nomask (bool):
            If True, skip computing losses over unmasked frames.

            This option corresponds to ``skip_nomask`` from ``fairseq``.

        num_classes (int):
            The number of classes in the labels.

        final_dim (int):
            Project final representations and targets to `final_dim`.

            This option corresponds to ``final_dim`` from ``fairseq``.

    Returns:
        HuBERTPretrainModel:
            The resulting model.
    """  # noqa: E501
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias
    )
    encoder = components._get_encoder(
        in_features=extractor_conv_layer_config[-1][0],
        embed_dim=encoder_embed_dim,
        dropout_input=encoder_projection_dropout,
        pos_conv_kernel=encoder_pos_conv_kernel,
        pos_conv_groups=encoder_pos_conv_groups,
        num_layers=encoder_num_layers,
        num_heads=encoder_num_heads,
        attention_dropout=encoder_attention_dropout,
        ff_interm_features=encoder_ff_interm_features,
        ff_interm_dropout=encoder_ff_interm_dropout,
        dropout=encoder_dropout,
        layer_norm_first=encoder_layer_norm_first,
        layer_drop=encoder_layer_drop,
    )
    wav2vec2 = Wav2Vec2Model(feature_extractor, encoder)
    mask_generator = components.MaskGenerator(
        encoder_embed_dim,
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
    logit_generator = components.LogitGenerator(
        encoder_embed_dim,
        num_classes,
        final_dim,
        skip_masked,
        skip_nomask,
    )
    return HuBERTPretrainModel(wav2vec2=wav2vec2, mask_generator=mask_generator, logit_generator=logit_generator)


def hubert_pretrain_base(
    encoder_projection_dropout: float = 0.1,
    encoder_attention_dropout: float = 0.1,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.1,
    encoder_layer_drop: float = 0.05,
    num_classes: int = 100,
) -> HuBERTPretrainModel:
    # Overriding the signature so that the return type is correct on Sphinx
    """hubert_pretrain_base(encoder_projection_dropout: float = 0.1, encoder_attention_dropout: float = 0.1, encoder_ff_interm_dropout: float = 0.0, encoder_dropout: float = 0.1, encoder_layer_drop: float = 0.05, num_classes: int = 100) -> torchaudio.models.HuBERTPretrainModel

    Build HuBERTPretrainModel model with "base" architecture from *HuBERT* [:footcite:`hsu2021hubert`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_attention_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_layer_drop (float):
            See :py:func:`hubert_pretrain_model`.
        num_classes (int, optional):
            See :py:func:`hubert_pretrain_model`.

    Returns:
        HuBERTPretrainModel:
            The resulting model.
    """  # noqa: E501
    return hubert_pretrain_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=False,
        encoder_layer_drop=encoder_layer_drop,
        mask_prob=0.80,
        mask_selection="static",
        mask_other=0.0,
        mask_length=10,
        no_mask_overlap=False,
        mask_min_space=1,
        mask_channel_prob=0.0,
        mask_channel_selection="static",
        mask_channel_other=0.0,
        mask_channel_length=10,
        no_mask_channel_overlap=False,
        mask_channel_min_space=1,
        skip_masked=False,
        skip_nomask=False,
        num_classes=num_classes,
        final_dim=256,
    )


def hubert_pretrain_large(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
) -> HuBERTPretrainModel:
    # Overriding the signature so that the return type is correct on Sphinx
    """hubert_pretrain_large(encoder_projection_dropout: float = 0.0, encoder_attention_dropout: float = 0.0, encoder_ff_interm_dropout: float = 0.0, encoder_dropout: float = 0.0, encoder_layer_drop: float = 0.0) -> torchaudio.models.HuBERTPretrainModel

    Build HuBERTPretrainModel model for pre-training with "large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_attention_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_layer_drop (float):
            See :py:func:`hubert_pretrain_model`.

    Returns:
        HuBERTPretrainModel:
            The resulting model.
    """  # noqa: E501
    return hubert_pretrain_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        mask_prob=0.80,
        mask_selection="static",
        mask_other=0.0,
        mask_length=10,
        no_mask_overlap=False,
        mask_min_space=1,
        mask_channel_prob=0.0,
        mask_channel_selection="static",
        mask_channel_other=0.0,
        mask_channel_length=10,
        no_mask_channel_overlap=False,
        mask_channel_min_space=1,
        skip_masked=False,
        skip_nomask=False,
        num_classes=500,
        final_dim=768,
    )


def hubert_pretrain_xlarge(
    encoder_projection_dropout: float = 0.0,
    encoder_attention_dropout: float = 0.0,
    encoder_ff_interm_dropout: float = 0.0,
    encoder_dropout: float = 0.0,
    encoder_layer_drop: float = 0.0,
) -> HuBERTPretrainModel:
    # Overriding the signature so that the return type is correct on Sphinx
    """hubert_pretrain_xlarge(encoder_projection_dropout: float = 0.0, encoder_attention_dropout: float = 0.0, encoder_ff_interm_dropout: float = 0.0, encoder_dropout: float = 0.0, encoder_layer_drop: float = 0.0) -> torchaudio.models.HuBERTPretrainModel

    Build HuBERTPretrainModel model for pre-training with "extra large" architecture from *HuBERT* [:footcite:`hsu2021hubert`]

    Args:
        encoder_projection_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_attention_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_ff_interm_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_dropout (float):
            See :py:func:`hubert_pretrain_model`.
        encoder_layer_drop (float):
            See :py:func:`hubert_pretrain_model`.

    Returns:
        HuBERTPretrainModel:
            The resulting model.
    """  # noqa: E501
    return hubert_pretrain_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1280,
        encoder_projection_dropout=encoder_projection_dropout,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=48,
        encoder_num_heads=16,
        encoder_attention_dropout=encoder_attention_dropout,
        encoder_ff_interm_features=5120,
        encoder_ff_interm_dropout=encoder_ff_interm_dropout,
        encoder_dropout=encoder_dropout,
        encoder_layer_norm_first=True,
        encoder_layer_drop=encoder_layer_drop,
        mask_prob=0.80,
        mask_selection="static",
        mask_other=0.0,
        mask_length=10,
        no_mask_overlap=False,
        mask_min_space=1,
        mask_channel_prob=0.0,
        mask_channel_selection="static",
        mask_channel_other=0.0,
        mask_channel_length=10,
        no_mask_channel_overlap=False,
        mask_channel_min_space=1,
        skip_masked=False,
        skip_nomask=False,
        num_classes=500,
        final_dim=1024,
    )
