"""Import fariseq's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `fairseq`.
"""
import re

from torch.nn import Module

from ..model import Wav2Vec2Model, _get_model


def _parse_config(original):
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


def _map_group(k):
    # "conv_layers.X.0.weight" -> "conv_layers.X.conv.weight"
    # "conv_layers.0.2.weight" -> "conv_layers.0.layer_norm.weight"
    # "conv_layers.0.2.bias"   -> "conv_layers.0.layer_norm.bias"
    p0 = r"conv_layers\.0\.2\.(weight|bias)"
    p1 = r"conv_layers\.(\d+)\.0\.weight"
    match = re.match(p0, k)
    if match:
        return f"conv_layers.0.layer_norm.{match.group(1)}"
    match = re.match(p1, k)
    if match:
        return f"conv_layers.{match.group(1)}.conv.weight"
    raise ValueError(f'Unexpected key: {k}')


def _map_layer(k):
    # "conv_layers.X.0.weight"   -> "conv_layers.X.conv.weight"
    # "conv_layers.X.0.bias"     -> "conv_layers.X.conv.bias"
    # "conv_layers.X.2.1.weight" -> "conv_layers.X.layer_norm.weight"
    # "conv_layers.X.2.1.bias"   -> "conv_layers.X.layer_norm.bias"
    p0 = r"conv_layers\.(\d+)\.0\.(weight|bias)"
    p1 = r"conv_layers\.(\d+)\.2\.1\.(weight|bias)"
    match = re.match(p0, k)
    if match:
        return f"conv_layers.{match.group(1)}.conv.{match.group(2)}"
    match = re.match(p1, k)
    if match:
        return f"conv_layers.{match.group(1)}.layer_norm.{match.group(2)}"
    raise ValueError(f'Unexpected key: {k}')


def _map_extractor_key(key, mode):
    _map = _map_group if mode == "group_norm" else _map_layer
    return _map(key)


def _map_keys(state_dict, extractor_mode):
    mapped = {}
    # feature Extractor
    extractor = 'w2v_model.feature_extractor.'
    # feature projection
    proj = 'w2v_model.post_extract_proj.'
    proj_layer = 'w2v_model.layer_norm.'
    # positional embedding
    pos_conv = 'w2v_model.encoder.pos_conv.0.'
    pos_conv_norm = 'w2v_model.encoder.layer_norm.'
    # encoder layer
    enc_layers = 'w2v_model.encoder.layers.'
    for k, v in state_dict.items():
        _k = k
        if k == 'w2v_model.mask_emb':
            continue
        if k.startswith(extractor):
            k = f"feature_extractor.{_map_extractor_key(k.replace(extractor, ''), extractor_mode)}"
        elif k.startswith(proj_layer):
            k = f"encoder.feature_projection.layer_norm.{k.replace(proj_layer, '')}"
        elif k.startswith(proj):
            k = f"encoder.feature_projection.projection.{k.replace(proj, '')}"
        elif k.startswith(pos_conv):
            k = f"encoder.transformer.pos_conv_embed.conv.{k.replace(pos_conv, '')}"
        elif k.startswith(pos_conv_norm):
            k = f"encoder.transformer.layer_norm.{k.replace(pos_conv_norm, '')}"
        elif k.startswith(enc_layers):
            k = f"{k.replace(enc_layers, '')}"
            i, k = k.split('.', 1)
            if k.startswith('self_attn_layer_norm.'):
                k = f"encoder.transformer.layers.{i}.layer_norm.{k.replace('self_attn_layer_norm.', '')}"
            elif k.startswith('self_attn.'):
                k = f"encoder.transformer.layers.{i}.attention.{k.replace('self_attn.', '')}"
            elif k.startswith('fc1.'):
                k = f"encoder.transformer.layers.{i}.feed_forward.intermediate_dense.{k.replace('fc1.', '')}"
            elif k.startswith('fc2.'):
                k = f"encoder.transformer.layers.{i}.feed_forward.output_dense.{k.replace('fc2.', '')}"
            elif k.startswith('final_layer_norm.'):
                k = f"encoder.transformer.layers.{i}.final_layer_norm.{k.replace('final_layer_norm.', '')}"
            else:
                raise ValueError(f'Unexpected key: {_k}')
        elif k.startswith('proj.'):
            k = f"encoder.readout.{k.replace('proj.', '')}"
        else:
            raise ValueError(f'Unexpected key: {_k}')
        mapped[k] = v
    return mapped


def import_fairseq_finetuned_model(original: Module) -> Wav2Vec2Model:
    """Import finetuned wav2vec2 mdoel from `fairseq`_.

    Args:
        model (Wav2VecEncoder):
            An instance of ``fairseq.models.wav2vec.wav2vec2_asr.Wav2VecEncoder``.
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
    model = _get_model(**config)
    model.load_state_dict(_map_keys(original.state_dict(), config['extractor_mode']))
    return model
