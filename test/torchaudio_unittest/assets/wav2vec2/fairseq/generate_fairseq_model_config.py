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
    _this_dir = os.path.dirname(os.path.abspath(__file__))

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
        default=_this_dir,
        help='Directory where `dict.ltr.txt` file is found.'
    )
    return parser.parse_args()


def _to_json(conf):
    import yaml
    from omegaconf import OmegaConf
    return yaml.safe_load(OmegaConf.to_yaml(conf))


def _load(model_file, dict_dir):
    import fairseq

    overrides = {'data': dict_dir}
    model, args, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_file], arg_overrides=overrides
    )
    model = model[0]
    return model, _to_json(args), _to_json(task.cfg)


def _main():
    args = _parse_args()
    _, args, task_cfg = _load(args.model_file, args.dict_dir)

    conf = {
        'model': args['model'],
        'task': task_cfg,
    }

    for key in ['model', 'task']:
        conf[key]['data'] = '<PATH_TO_ASSET_DIRECTORY_WHICH_HAS_TO_BE_UPDATED_AT_RUNTIME>'

    print(json.dumps(conf, indent=4, sort_keys=True))


if __name__ == '__main__':
    _main()
