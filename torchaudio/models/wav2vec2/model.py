from typing import Optional, Tuple, List

from torch import Tensor
from torch.nn import Module

from . import components


class Wav2Vec2Model(Module):
    """Encoder model used in *wav2vec 2.0* [:footcite:`baevski2020wav2vec`].

    Note:
        To build the model, please use one of the factory functions.

    Args:
        feature_extractor (torch.nn.Module):
            Feature extractor that extracts feature vectors from raw audio Tensor.

        encoder (torch.nn.Module):
            Encoder that converts the audio features into the sequence of probability
            distribution (in negative log-likelihood) over labels.
    """
    def __init__(
            self,
            feature_extractor: Module,
            encoder: Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder

    def extract_features(
            self,
            waveforms: Tensor,
            lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Extract feature vectors from raw waveforms

        Args:
            waveforms (Tensor): Audio tensor of shape ``(batch, frames)``.
            lengths (Tensor, optional):
                Indicates the valid length of each audio sample in the batch.
                Shape: ``(batch, )``.

        Returns:
            Tensor:
                Feature vectors.
                Shape: ``(batch, frames, feature dimention)``
            Tensor, optional:
                Indicates the valid length of each feature in the batch, computed
                based on the given ``lengths`` argument.
                Shape: ``(batch, )``.
        """
        return self.feature_extractor(waveforms, lengths)

    def forward(
            self,
            waveforms: Tensor,
            lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute the sequence of probability distribution over labels.

        Args:
            waveforms (Tensor): Audio tensor of shape ``(batch, frames)``.
            lengths (Tensor, optional):
                Indicates the valid length of each audio sample in the batch.
                Shape: ``(batch, )``.

        Returns:
            Tensor:
                The sequences of probability distribution (in logit) over labels.
                Shape: ``(batch, frames, num labels)``.
            Tensor, optional:
                Indicates the valid length of each feature in the batch, computed
                based on the given ``lengths`` argument.
                Shape: ``(batch, )``.
        """
        x, lengths = self.feature_extractor(waveforms, lengths)
        return self.encoder(x, lengths), lengths


def _get_model(
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
        encoder_num_out: int,
) -> Wav2Vec2Model:
    if extractor_conv_layer_config is None:
        extractor_conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    feature_extractor = components._get_feature_extractor(
        extractor_mode, extractor_conv_layer_config, extractor_conv_bias)
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
        num_out=encoder_num_out,
    )
    return Wav2Vec2Model(feature_extractor, encoder)


def wav2vec2_base(num_out: int) -> Wav2Vec2Model:
    """Build wav2vec2.0 model with "Base" configuration from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`].

    Args:
        num_out: int
            The number of output labels.

    Returns:
        Wav2Vec2Model: The resulting model.

    Example - Reload fine-tuned model from Hugging Face:
        >>> # Session 1 - Convert pretrained model from Hugging Face and save the parameters.
        >>> from torchaudio.models.wav2vec2.utils import import_huggingface_model
        >>>
        >>> original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = import_huggingface_model(original)
        >>> torch.save(model.state_dict(), "wav2vec2-base-960h.pt")
        >>>
        >>> # Session 2 - Load model and the parameters
        >>> model = wav2vec2_base(num_out=32)
        >>> model.load_state_dict(torch.load("wav2vec2-base-960h.pt"))
    """
    return _get_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=768,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=12,
        encoder_num_heads=12,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=3072,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.1,
        encoder_layer_norm_first=False,
        encoder_layer_drop=0.1,
        encoder_num_out=num_out,
    )


def wav2vec2_large(num_out: int) -> Wav2Vec2Model:
    """Build wav2vec2.0 model with "Large" configuration from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`].

    Args:
        num_out: int
            The number of output labels.

    Returns:
        Wav2Vec2Model: The resulting model.

    Example - Reload fine-tuned model from Hugging Face:
        >>> # Session 1 - Convert pretrained model from Hugging Face and save the parameters.
        >>> from torchaudio.models.wav2vec2.utils import import_huggingface_model
        >>>
        >>> original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        >>> model = import_huggingface_model(original)
        >>> torch.save(model.state_dict(), "wav2vec2-base-960h.pt")
        >>>
        >>> # Session 2 - Load model and the parameters
        >>> model = wav2vec2_large(num_out=32)
        >>> model.load_state_dict(torch.load("wav2vec2-base-960h.pt"))
    """
    return _get_model(
        extractor_mode="group_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=False,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.1,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.1,
        encoder_layer_norm_first=False,
        encoder_layer_drop=0.1,
        encoder_num_out=num_out,
    )


def wav2vec2_large_lv60k(num_out: int) -> Wav2Vec2Model:
    """Build wav2vec2.0 model with "Large LV-60k" configuration from *wav2vec 2.0* [:footcite:`baevski2020wav2vec`].

    Args:
        num_out: int
            The number of output labels.

    Returns:
        Wav2Vec2Model: The resulting model.

    Example - Reload fine-tuned model from Hugging Face:
        >>> # Session 1 - Convert pretrained model from Hugging Face and save the parameters.
        >>> from torchaudio.models.wav2vec2.utils import import_huggingface_model
        >>>
        >>> original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        >>> model = import_huggingface_model(original)
        >>> torch.save(model.state_dict(), "wav2vec2-base-960h.pt")
        >>>
        >>> # Session 2 - Load model and the parameters
        >>> model = wav2vec2_large_lv60k(num_out=32)
        >>> model.load_state_dict(torch.load("wav2vec2-base-960h.pt"))
    """
    return _get_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=None,
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.1,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        encoder_num_out=num_out,
    )
