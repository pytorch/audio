# Emformer RNN-T ASR Example

This directory contains sample implementations of training and evaluation pipelines for an on-device-oriented streaming-capable Emformer RNN-T ASR model.

## Usage

### Training

[`train.py`](./train.py) trains an Emformer RNN-T model on LibriSpeech using PyTorch Lightning. Note that the script expects users to have access to GPU nodes for training and provide paths to the full LibriSpeech dataset and the SentencePiece model to be used to encode targets. The script also expects a file (--global_stats_path) that contains training set feature statistics; this file can be generated via [`global_stats.py`](./global_stats.py).

Sample SLURM command:
```
srun --cpus-per-task=12 --gpus-per-node=8 -N 4 --ntasks-per-node=8 python train.py --exp_dir ./experiments --librispeech_path ./librispeech/ --global_stats_path ./global_stats.json --sp_model_path ./spm_bpe_4096.model
```

### Evaluation

[`eval.py`](./eval.py) evaluates a trained Emformer RNN-T model on LibriSpeech test-clean.

Using the default configuration along with a SentencePiece model trained on LibriSpeech with vocab size 4096 and type bpe, [`train.py`](./train.py) produces a model with 76.7M parameters (307MB) that achieves an WER of 0.0466 when evaluated on test-clean with [`eval.py`](./eval.py).

The table below contains WER results for various splits.

|                     |          WER |
|:-------------------:|-------------:|
| test-clean          |       0.0456 |
| test-other          |       0.1066 |
| dev-clean           |       0.0415 |
| dev-other           |       0.1110 |


Sample SLURM command:
```
srun python eval.py --checkpoint_path ./experiments/checkpoints/epoch=119-step=208079.ckpt --librispeech_path ./librispeech/ --sp_model_path ./spm_bpe_4096.model --use_cuda
```
