#!/usr/bin/env python3
"""Convert a Wav2Vec2/HuBERT model published by fairseq into torchaudio format

Examples

```
# wav2vec2
python build_tools/convert_fairseq_models.py \
  --input-file wav2vec_small.pt \
  --output-file wav2vec2_fairseq_base_ls960.pth

python build_tools/convert_fairseq_models.py \
  --input-file wav2vec_small_10m.pt \
  --out wav2vec2_fairseq_base_ls960_asr_ll10m.pth \
  --dict <dict-dir>

python build_tools/convert_fairseq_models.py \
  --input-file wav2vec_small_100h.pt \
  --output-file wav2vec2_fairseq_base_ls960_asr_ls100h.pth \
  --dict <dict-dir>

# HuBERT
python convert_fairseq_models.py \
  --input-file hubert_base_ls960.pt \
  --output-file hubert_fairseq_base_ls960.pth

python convert_fairseq_models.py \
  --input-file hubert_large_ll60k.pt \
  --output-file hubert_fairseq_large_ll60k.pth

python convert_fairseq_models.py \
  --input-file hubert_large_ll60k_finetune_ls960.pt \
  --output-file hubert_fairseq_large_ll60k_asr_ls960.pth

python convert_fairseq_models.py \
  --input-file hubert_xtralarge_ll60k.pt \
  --output-file hubert_fairseq_xlarge_ll60k.pth

python convert_fairseq_models.py \
  --input-file hubert_xtralarge_ll60k_finetune_ls960.pt \
  --output-file hubert_fairseq_xlarge_ll60k_asr_ls960.pth
```
"""

import argparse

# Note: Avoiding the import of torch and fairseq on global scope as they are slow


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--input-file', required=True,
        help='Input model file.'
    )
    parser.add_argument(
        '--output-file', required=False,
        help='Output model file.'
    )
    parser.add_argument(
        '--dict-dir',
        help=(
            'Directory where letter vocabulary file, `dict.ltr.txt`, is found. '
            'Required when loading wav2vec2 model. '
            'https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt'
        )
    )
    return parser.parse_args()


def _load_model(input_file, dict_dir):
    import fairseq

    overrides = {} if dict_dir is None else {'data': dict_dir}
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [input_file], arg_overrides=overrides,
    )
    return models[0]


def _import_model(model):
    from torchaudio.models.wav2vec2.utils import import_fairseq_model

    if model.__class__.__name__ in ['Wav2VecCtc', 'HubertCtc']:
        model = model.w2v_encoder
    model = import_fairseq_model(model)
    return model


def _main(args):
    import torch
    model = _load_model(args.input_file, args.dict_dir)
    model = _import_model(model)
    torch.save(model.state_dict(), args.output_file)


if __name__ == '__main__':
    _main(_parse_args())
