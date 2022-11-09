"""Import Hugging Face transformers's wav2vec2.0 pretrained weights to torchaudios's format.
"""
import logging

from torch.nn import Module

from ..model import wav2vec2_model, Wav2Vec2Model, wavlm_model

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


def _get_config_wavlm(cfg):
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
        "encoder_num_buckets": cfg.num_buckets,
        "encoder_max_distance": cfg.max_bucket_distance,
        "encoder_attention_dropout": cfg.attention_dropout,
        "encoder_ff_interm_features": cfg.intermediate_size,
        "encoder_ff_interm_dropout": cfg.activation_dropout,
        "encoder_dropout": cfg.hidden_dropout,
        "encoder_layer_norm_first": cfg.do_stable_layer_norm,
        "encoder_layer_drop": cfg.layerdrop,
    }
    return config


def _build(config, original):
    is_for_ctc = original.__class__.__name__ in ["Wav2Vec2ForCTC", "WavLMForCTC"]
    if is_for_ctc:
        aux_num_out = original.config.vocab_size
        wav2vec2 = original.wav2vec2
    else:
        _LG.warning(
            "The model is not an instance of Wav2Vec2ForCTC or WavLMForCTC. " '"lm_head" module is not imported.'
        )
        aux_num_out = None
        wav2vec2 = original
    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    if is_wavlm:
        imported = wavlm_model(**config, aux_num_out=aux_num_out)
    else:
        imported = wav2vec2_model(**config, aux_num_out=aux_num_out)
    imported.feature_extractor.load_state_dict(wav2vec2.feature_extractor.state_dict())
    imported.encoder.feature_projection.load_state_dict(wav2vec2.feature_projection.state_dict())
    encoder_state_dict = wav2vec2.encoder.state_dict()
    if is_wavlm:  # Rename paramaters of linear transformations for compatibility with the HF model
        encoder_state_dict = {rename_wavlm_key(x): encoder_state_dict[x] for x in encoder_state_dict.keys()}
    imported.encoder.transformer.load_state_dict(encoder_state_dict)
    if is_for_ctc:
        imported.aux.load_state_dict(original.lm_head.state_dict())
    return imported


def rename_wavlm_key(key):
    """Rename weights and biases of linear transformations, since we define them directly in WavLMSelfAttention,
    as opposed to nesting them in Linear modules
    """
    return (
        key.replace("k_proj.weight", "k_proj_weight")
        .replace("k_proj.bias", "k_proj_bias")
        .replace("q_proj.weight", "q_proj_weight")
        .replace("q_proj.bias", "q_proj_bias")
        .replace("v_proj.weight", "v_proj_weight")
        .replace("v_proj.bias", "v_proj_bias")
        .replace("out_proj.weight", "out_proj_weight")
        .replace("out_proj.bias", "out_proj_bias")
    )


def import_huggingface_model(original: Module) -> Wav2Vec2Model:
    """Builds :class:`Wav2Vec2Model` from the corresponding model object of
    `Transformers <https://huggingface.co/transformers/>`_.

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
    """
    _LG.info("Importing model.")
    _LG.info("Loading model configuration.")
    is_wavlm = original.__class__.__name__ in ["WavLMModel", "WavLMForCTC"]
    if is_wavlm:
        config = _get_config_wavlm(original.config)
    else:
        config = _get_config(original.config)
    _LG.debug("  - config: %s", config)
    _LG.info("Building model.")
    imported = _build(config, original)
    return imported
