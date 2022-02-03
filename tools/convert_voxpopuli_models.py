#!/usr/bin/env python3
"""Convert the fairseq models available in voxpopuli repo https://github.com/facebookresearch/voxpopuli

The available checkpoints should open with fairseq.
But the following error cannot be resolved with almost any version of fairseq.
https://github.com/facebookresearch/voxpopuli/issues/29

So this script manually parse the checkpoint file and reconstruct the model.

Examples

```
python convert_voxpopuli_models.py \
  --input-file wav2vec2_base_10k_ft_fr.pt \
  --output-file wav2vec2_voxpopuli_base_10k_asr_fr.pt
```
"""


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input-file", required=True, help="Input checkpoint file.")
    parser.add_argument("--output-file", required=False, help="Output model file.")
    return parser.parse_args()


def _removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def _load(input_file):
    import torch
    from omegaconf import OmegaConf

    data = torch.load(input_file)
    cfg = OmegaConf.to_container(data["cfg"])
    for key in list(cfg.keys()):
        if key != "model":
            del cfg[key]
            if "w2v_args" in cfg["model"]:
                del cfg["model"]["w2v_args"][key]
    state_dict = {_removeprefix(k, "w2v_encoder."): v for k, v in data["model"].items()}
    return cfg, state_dict


def _parse_model_param(cfg, state_dict):
    key_mapping = {
        "extractor_mode": "extractor_mode",
        "conv_feature_layers": "extractor_conv_layer_config",
        "conv_bias": "extractor_conv_bias",
        "encoder_embed_dim": "encoder_embed_dim",
        "dropout_input": "encoder_projection_dropout",
        "conv_pos": "encoder_pos_conv_kernel",
        "conv_pos_groups": "encoder_pos_conv_groups",
        "encoder_layers": "encoder_num_layers",
        "encoder_attention_heads": "encoder_num_heads",
        "attention_dropout": "encoder_attention_dropout",
        "encoder_ffn_embed_dim": "encoder_ff_interm_features",
        "activation_dropout": "encoder_ff_interm_dropout",
        "dropout": "encoder_dropout",
        "layer_norm_first": "encoder_layer_norm_first",
        "layerdrop": "encoder_layer_drop",
        "encoder_layerdrop": "encoder_layer_drop",
    }
    params = {}
    src_dicts = [cfg["model"]]
    if "w2v_args" in cfg["model"]:
        src_dicts.append(cfg["model"]["w2v_args"]["model"])

    for src, tgt in key_mapping.items():
        for model_cfg in src_dicts:
            if src in model_cfg:
                params[tgt] = model_cfg[src]
                break
    if params["extractor_mode"] == "default":
        params["extractor_mode"] = "group_norm"
    # the following line is commented out to resolve lint warning; uncomment before running script
    # params["extractor_conv_layer_config"] = eval(params["extractor_conv_layer_config"])
    assert len(params) == 15
    params["aux_num_out"] = state_dict["proj.bias"].numel() if "proj.bias" in state_dict else None
    return params


def _main(args):
    import json

    import torch
    import torchaudio
    from torchaudio.models.wav2vec2.utils.import_fairseq import _convert_state_dict as _convert

    cfg, state_dict = _load(args.input_file)
    params = _parse_model_param(cfg, state_dict)
    print(json.dumps(params, indent=4))
    model = torchaudio.models.wav2vec2_model(**params)
    model.load_state_dict(_convert(state_dict))
    torch.save(model.state_dict(), args.output_file)


if __name__ == "__main__":
    _main(_parse_args())
