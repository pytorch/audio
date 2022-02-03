"""Import Hugging Face transformers's wav2vec2.0 pretrained weights to torchaudios's format.
"""
import logging

from torch.nn import Module

from ..model import Wav2Vec2Model, wav2vec2_model

_LG = logging.getLogger(__name__)


def _get_config(cfg):
    config = {
        "extractor_mode": f"{cfg.feat_extract_norm}_norm",
        "extractor_conv_layer_config": list(zip(cfg.conv_dim, cfg.conv_kernel, cfg.conv_stride)),
        "extractor_conv_bias": cfg.conv_bias,
        "encoder_embed_dim": cfg.hidden_size,
        "encoder_projection_dropout": cfg.feat_proj_dropout,
        "encoder_pos_conv_kernel": cfg.num_conv_pos_embeddings,
        "encoder_pos_conv_groups": cfg.num_conv_pos_embedding_groups,
        "encoder_num_layers": cfg.num_hidden_layers,
        "encoder_num_heads": cfg.num_attention_heads,
        "encoder_attention_dropout": cfg.attention_dropout,
        "encoder_ff_interm_features": cfg.intermediate_size,
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        "encoder_dropout": cfg.hidden_dropout,
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        "encoder_layer_drop": cfg.layerdrop,
    }
    return config


def _build(config, original):
    if original.__class__.__name__ == "Wav2Vec2ForCTC":
        aux_num_out = original.config.vocab_size
        wav2vec2 = original.wav2vec2
    else:
        _LG.warning("The model is not an instance of Wav2Vec2ForCTC. " '"lm_head" module is not imported.')
        aux_num_out = None
        wav2vec2 = original
    imported = wav2vec2_model(**config, aux_num_out=aux_num_out)
    imported.feature_extractor.load_state_dict(wav2vec2.feature_extractor.state_dict())
    imported.encoder.feature_projection.load_state_dict(wav2vec2.feature_projection.state_dict())
    imported.encoder.transformer.load_state_dict(wav2vec2.encoder.state_dict())
    if original.__class__.__name__ == "Wav2Vec2ForCTC":
        imported.aux.load_state_dict(original.lm_head.state_dict())
    return imported


def import_huggingface_model(original: Module) -> Wav2Vec2Model:
    """import_huggingface_model(original: torch.nn.Module) -> torchaudio.models.Wav2Vec2Model

    Build Wav2Vec2Model from the corresponding model object of Hugging Face's `Transformers`_.

    Args:
        original (torch.nn.Module): An instance of ``Wav2Vec2ForCTC`` from ``transformers``.

    Returns:
        Wav2Vec2Model: Imported model.

    Example
        >>> from torchaudio.models.wav2vec2.utils import import_huggingface_model
        >>>
        >>> original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = import_huggingface_model(original)
        >>>
        >>> waveforms, _ = torchaudio.load("audio.wav")
        >>> logits, _ = model(waveforms)

    .. _Transformers: https://huggingface.co/transformers/
    """
    _LG.info("Importing model.")
    _LG.info("Loading model configuration.")
    config = _get_config(original.config)
    _LG.debug("  - config: %s", config)
    _LG.info("Building model.")
    imported = _build(config, original)
    return imported
