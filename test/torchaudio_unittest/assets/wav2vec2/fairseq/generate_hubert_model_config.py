#!/usr/bin/env python3
"""Generate the conf JSONs from fairseq pretrained weight file, consumed by unit tests

Note:
    The current configuration files were generated on fairseq e47a4c84

Usage:
1. Download pretrained parameters from https://github.com/pytorch/fairseq/tree/main/examples/hubert
2. Run this script and save the resulting JSON configuration in assets directory.

Example:

```
python generate_hubert_model_config.py \
    --model-file hubert_base_ls960.pt \
    > hubert_base_ls960.json

python generate_hubert_model_config.py \
    --model-file hubert_large_ll60k.pt \
    > hubert_large_ll60k.json

python generate_hubert_model_config.py \
    --model-file hubert_large_ll60k_finetune_ls960.pt \
    > hubert_large_ll60k_finetune_ls960.json

python generate_hubert_model_config.py \
    --model-file hubert_xlarge_ll60k.pt \
    > hubert_large_ll60k.json

python generate_hubert_model_config.py \
    --model-file hubert_xlarge_ll60k_finetune_ls960.pt \
    > hubert_large_ll60k_finetune_ls960.json
```
"""
import argparse
import json


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-file",
        required=True,
        help=("A pt file from " "https://github.com/pytorch/fairseq/tree/main/examples/hubert"),
    )
    return parser.parse_args()


def _load(model_file):
    import fairseq
    from omegaconf import OmegaConf

    models, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    model = models[0]
    cfg = OmegaConf.to_container(cfg)
    return model, cfg


def _main():
    args = _parse_args()
    model, cfg = _load(args.model_file)

    if model.__class__.__name__ == "HubertModel":
        cfg["task"]["data"] = "/foo/bar"
        cfg["task"]["label_dir"] = None
        conf = {
            "_name": "hubert",
            "model": cfg["model"],
            "task": cfg["task"],
            "num_classes": model.num_classes,
        }
    elif model.__class__.__name__ == "HubertCtc":
        conf = cfg["model"]
        del conf["w2v_path"]
        keep = ["_name", "task", "model"]
        for key in conf["w2v_args"]:
            if key not in keep:
                del conf["w2v_args"][key]
        conf["data"] = "/foo/bar/"
        conf["w2v_args"]["task"]["data"] = "/foo/bar"
        conf["w2v_args"]["task"]["labels"] = []
        conf["w2v_args"]["task"]["label_dir"] = "/foo/bar"
    print(json.dumps(conf, indent=4, sort_keys=True))


if __name__ == "__main__":
    _main()
