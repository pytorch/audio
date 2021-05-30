#!/usr/bin/env python3
"""Generate the conf JSON from fairseq pretrained weight file, that is consumed by unit tests

Example usage:

```
python generate_fairseq_model_config.py \
    --model-file wav2vec_small_960h.pt \
    > wav2vec_small_960h.json

python generate_fairseq_model_config.py \
    --model-file wav2vec_big_960h.pt \
    > wav2vec_large_960h.json

python generate_fairseq_model_config.py \
    --model-file wav2vec2_vox_960h_new.pt \
    > wav2vec_large_lv60_960h.json

python generate_fairseq_model_config.py \
    --model-file wav2vec_vox_960h_pl.pt \
    > wav2vec_large_lv60_self_960h.json
```
"""
import os
import json
import argparse


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--model-file',
        required=True,
        help=(
            'Fine-tuned Model check point file available from '
            'https://github.com/pytorch/fairseq/tree/master/examples/wav2vec'
        )
    )
    parser.add_argument(
        '--dict-dir',
        help=(
            'Directory where `dict.ltr.txt` file is found. '
            'Default: the directory of the given model.'
        )
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

    overrides = {'data': dict_dir}
    _, args, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_file], arg_overrides=overrides
    )
    return _to_json(args)


def _main():
    args = _parse_args()
    conf = _load(args.model_file, args.dict_dir)

    del conf['model']['data']
    del conf['model']['w2v_args']['task']['data']

    print(json.dumps(conf['model'], indent=4, sort_keys=True))


if __name__ == '__main__':
    _main()
