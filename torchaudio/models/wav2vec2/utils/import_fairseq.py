"""Import fariseq's wav2vec2.0 pretrained weights to torchaudios's format.

For this module to work, you need `fairseq`.
"""
import re

from torch.nn import Module

from ..model import Wav2Vec2Model, _get_model


def _parse_args(original, args):
    finetune = args['model']
    pretrain = args['model']['w2v_args']['model']
    # "layerdrop" and "activation_dropout" are overwritten in fine tune configuration.
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/base_960h.yaml#L54-L55
    # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/examples/wav2vec/config/finetuning/vox_960h.yaml#L54-L55
    config = {
        'extractor_mode': "group_norm" if pretrain['extractor_mode'] == 'default' else "layer_norm",
        'extractor_conv_layer_config': eval(pretrain['conv_feature_layers']),
        # Unfortunately, convolution layer feature is in string format, so we need to use `eval`.
        # JSON parser does not work here.
        # https://github.com/pytorch/fairseq/blob/425c36eafff535fe7337f8bdd5ace22ebacc78cb/fairseq/models/wav2vec/wav2vec2.py#L94-L100
        'extractor_conv_bias': pretrain['conv_bias'],
        'encoder_embed_dim': pretrain['encoder_embed_dim'],
        'encoder_projection_dropout': pretrain['dropout_input'],
        'encoder_pos_conv_kernel': pretrain['conv_pos'],
        'encoder_pos_conv_groups': pretrain['conv_pos_groups'],
        'encoder_num_layers': pretrain['encoder_layers'],
        'encoder_num_heads': pretrain['encoder_attention_heads'],
        'encoder_attention_dropout': pretrain['attention_dropout'],
        'encoder_ff_interm_features': pretrain['encoder_ffn_embed_dim'],
        'encoder_ff_interm_dropout': finetune['activation_dropout'],
        'encoder_dropout': pretrain['dropout'],
        'encoder_layer_norm_first': pretrain['layer_norm_first'],
        'encoder_layer_drop': finetune['layerdrop'],
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


def _import_state_dict(imported, config, original):
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


def import_fairseq_finetuned_model(original: Module, args: dict) -> Wav2Vec2Model:
    """Import finetuned wav2vec2 mdoel from `fairseq`_.

    Args:
        model (Wav2VecCtc): Model loaded with
            ``fairseq.checkpoint_utils.load_model_ensemble_and_task``.
        args (dict): Arguments returned by
            ``fairseq.checkpoint_utils.load_model_ensemble_and_task``.
    Returns:
        Wav2Vec2Model:
            An instance of the corresponding model class.

    Example:
        >>> model, args, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ...     [checkpoint_path], arg_overrides={'data': data_dir})
        >>> imported = import_fairseq_model(model[0], args)

    .. _fairseq: https://github.com/pytorch/fairseq
    """
    config = _parse_args(original.w2v_encoder, args)
    model = _get_model(**config)
    _import_state_dict(model, config, original.w2v_encoder)
    return model
