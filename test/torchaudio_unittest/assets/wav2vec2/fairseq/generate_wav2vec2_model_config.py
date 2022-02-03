#!/usr/bin/env python3
"""Generate the conf JSON from fairseq pretrained weight file, that is consumed by unit tests

Usage:
1. Download pretrained parameters from https://github.com/pytorch/fairseq/tree/main/examples/wav2vec
2. Download the dict from https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt
   and put it in the same directory as parameter files.
3. Run this script and save the resulting JSON configuration in assets directory.

Example:

```
# Pretrained
python generate_wav2vec2_model_config.py \
    --model-file wav2vec_small.pt \
    > wav2vec_small.json

python generate_wav2vec2_model_config.py \
    --model-file libri960_big.pt \
    > libri960_big.json

python generate_wav2vec2_model_config.py \
    --model-file wav2vec_vox_new.pt \
    > wav2vec_vox_new.json

# Fine-tuned
python generate_wav2vec2_model_config.py \
    --model-file wav2vec_small_960h.pt \
    > wav2vec_small_960h.json

python generate_wav2vec2_model_config.py \
    --model-file wav2vec_big_960h.pt \
    > wav2vec_large_960h.json

python generate_wav2vec2_model_config.py \
    --model-file wav2vec2_vox_960h_new.pt \
    > wav2vec_large_lv60_960h.json

python generate_wav2vec2_model_config.py \
    --model-file wav2vec_vox_960h_pl.pt \
    > wav2vec_large_lv60_self_960h.json
```
"""
import argparse
import json
import os


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help=("A point file from " "https://github.com/pytorch/fairseq/tree/main/examples/wav2vec"),
    )
    parser.add_argument(
        "--dict-dir",
        help=("Directory where `dict.ltr.txt` file is found. " "Default: the directory of the given model."),
    )
    args = parser.parse_args()
    if args.dict_dir is None:
        args.dict_dir = os.path.dirname(args.model_file)
    return args


def _to_json(conf):
    import yaml
    from omegaconf import OmegaConf

    return yaml.safe_load(OmegaConf.to_yaml(conf))


def _load(model_file, dict_dir):
    import fairseq

    overrides = {"data": dict_dir}
    _, args, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file], arg_overrides=overrides)
    return _to_json(args["model"])


def _main():
    args = _parse_args()
    conf = _load(args.model_file, args.dict_dir)

    if conf["_name"] == "wav2vec_ctc":
        del conf["data"]
        del conf["w2v_args"]["task"]["data"]
        conf["w2v_args"] = {key: conf["w2v_args"][key] for key in ["model", "task"]}

    print(json.dumps(conf, indent=4, sort_keys=True))


if __name__ == "__main__":
    _main()
