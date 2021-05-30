"""Import fariseq's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `fairseq`.
"""
import re

from torch.nn import Module

from ..model import Wav2Vec2Model, _get_model


def _parse_config(original):
    print(original)

    w2v_model = original.w2v_model
    encoder = w2v_model.encoder
    conv_layers = w2v_model.feature_extractor.conv_layers

    extractor_mode = 'layer_norm'
    if 'GroupNorm' in conv_layers[0][2].__class__.__name__:
        extractor_mode = 'group_norm'
    else:
        extractor_mode = 'layer_norm'

    conv_layer_config = [(l[0].out_channels, l[0].kernel_size[0], l[0].stride[0]) for l in conv_layers]

    if all(l[0].bias is None for l in conv_layers):
        conv_bias = False
    elif all(l[0].bias is not None for l in conv_layers):
        conv_bias = True
    else:
        raise ValueError(
            'Either all the convolutions layers have bias term or none of them should.')

    config = {
        'extractor_mode': extractor_mode,
        'extractor_conv_layer_config': conv_layer_config,
        'extractor_conv_bias': conv_bias,
        'encoder_embed_dim': w2v_model.post_extract_proj.out_features,
        'encoder_projection_dropout': w2v_model.dropout_input.p,
        'encoder_pos_conv_kernel': encoder.pos_conv[0].kernel_size[0],
        'encoder_pos_conv_groups': encoder.pos_conv[0].groups,
        'encoder_num_layers': len(encoder.layers),
        'encoder_num_heads': encoder.layers[0].self_attn.num_heads,
        'encoder_attention_dropout': encoder.layers[0].self_attn.dropout_module.p,
        'encoder_ff_interm_features': encoder.layers[0].fc1.out_features,
        'encoder_ff_interm_dropout': encoder.layers[0].dropout2.p,
        'encoder_dropout': encoder.layers[0].dropout3.p,
        'encoder_layer_norm_first': encoder.layer_norm_first,
        'encoder_layer_drop': encoder.layerdrop,
        'encoder_num_out': original.proj.out_features,
    }
    return config


def _map_feature_extractor_state_dict(state_dict, mode):
    def _map_group(key):
        # - "conv_layers.X.0.weight" -> "conv_layers.X.conv.weight"
        # - "conv_layers.0.2.weight" -> "conv_layers.0.layer_norm.weight"
        # - "conv_layers.0.2.bias"   -> "conv_layers.0.layer_norm.bias"
        p0 = r"conv_layers\.0\.2\.(weight|bias)"
        p1 = r"conv_layers\.(\d+)\.0\.weight"
        match = re.match(p0, key)
        if match:
            return f"conv_layers.0.layer_norm.{match.group(1)}"
        match = re.match(p1, key)
        if match:
            return f"conv_layers.{match.group(1)}.conv.weight"
        raise ValueError(f'Unexpected key: {key}')

    def _map_layer(key):
        p0 = r"conv_layers\.(\d+)\.0\.(weight|bias)"
        p1 = r"conv_layers\.(\d+)\.2\.1\.(weight|bias)"
        match = re.match(p0, key)
        if match:
            return f"conv_layers.{match.group(1)}.conv.{match.group(2)}"
        match = re.match(p1, key)
        if match:
            return f"conv_layers.{match.group(1)}.layer_norm.{match.group(2)}"
        raise ValueError(f'Unexpected key: {key}')

    _map = _map_group if mode == "group_norm" else _map_layer
    return {_map(k): v for k, v in state_dict.items()}


def _copy(src, dst):
    dst.load_state_dict(src.state_dict())


def _build(config, original):
    imported = _get_model(**config)
    # Feature Extractor
    imported.feature_extractor.load_state_dict(
        _map_feature_extractor_state_dict(
            original.w2v_model.feature_extractor.state_dict(),
            config['extractor_mode'])
    )
    # Feature projection
    _copy(
        original.w2v_model.layer_norm,
        imported.encoder.feature_projection.layer_norm)
    _copy(
        original.w2v_model.post_extract_proj,
        imported.encoder.feature_projection.projection)
    # Transformer
    _copy(
        original.w2v_model.encoder.pos_conv[0],
        imported.encoder.transformer.pos_conv_embed.conv)
    _copy(
        original.w2v_model.encoder.layer_norm,
        imported.encoder.transformer.layer_norm)
    for imported_, original_ in zip(imported.encoder.transformer.layers, original.w2v_model.encoder.layers):
        _copy(original_.self_attn, imported_.attention)
        _copy(original_.self_attn_layer_norm, imported_.layer_norm)
        _copy(original_.fc1, imported_.feed_forward.intermediate_dense)
        _copy(original_.fc2, imported_.feed_forward.output_dense)
        _copy(original_.final_layer_norm, imported_.final_layer_norm)
    # Readout
    _copy(original.proj, imported.encoder.readout)
    return imported


def import_fairseq_finetuned_model(original: Module) -> Wav2Vec2Model:
    """Import finetuned wav2vec2 mdoel from `fairseq`_.

    Args:
        model (Wav2VecEncoder): An instance of ``fairseq.models.wav2vec.wav2vec2_asr.Wav2VecEncoder``.
    Returns:
        Wav2Vec2Model:
            An instance of the corresponding model class.

    Example:
        >>> model, args, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ...     [checkpoint_path], arg_overrides={'data': data_dir})
        >>> imported = import_fairseq_model(model[0].w2v_encoder)

    .. _fairseq: https://github.com/pytorch/fairseq
    """
    config = _parse_config(original)
    imported = _build(config, original)
    return imported
