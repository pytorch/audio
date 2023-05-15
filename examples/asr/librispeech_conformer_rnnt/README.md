# Conformer RNN-T ASR Example

This directory contains sample implementations of training and evaluation pipelines for a Conformer RNN-T ASR model.

## Setup
### Install PyTorch and TorchAudio nightly or from source
Because Conformer RNN-T is currently a prototype feature, you will need to either use the TorchAudio nightly build or build TorchAudio from source. Note also that GPU support is required for training.

To install the nightly, follow the directions at <https://pytorch.org/>.

To build TorchAudio from source, refer to the [contributing guidelines](https://github.com/pytorch/audio/blob/main/CONTRIBUTING.md).

### Install additional dependencies
```bash
pip install pytorch-lightning sentencepiece tensorboard
```

## Usage

### Training

[`train.py`](./train.py) trains an Conformer RNN-T model (30.2M parameters, 121MB) on LibriSpeech using PyTorch Lightning. Note that the script expects users to have the following:
- Access to GPU nodes for training.
- Full LibriSpeech dataset.
- SentencePiece model to be used to encode targets; the model can be generated using [`train_spm.py`](./train_spm.py).
- File (--global_stats_path) that contains training set feature statistics; this file can be generated using [`global_stats.py`](../emformer_rnnt/global_stats.py).

Sample SLURM command:
```
srun --cpus-per-task=12 --gpus-per-node=8 -N 4 --ntasks-per-node=8 python train.py --exp-dir ./experiments --librispeech-path ./librispeech/ --global-stats-path ./global_stats.json --sp-model-path ./spm_unigram_1023.model --epochs 160
```

### Evaluation

[`eval.py`](./eval.py) evaluates a trained Conformer RNN-T model on LibriSpeech test-clean.

Sample SLURM command:
```
srun python eval.py --checkpoint-path ./experiments/checkpoints/epoch=159.ckpt --librispeech-path ./librispeech/ --sp-model-path ./spm-unigram-1023.model --use-cuda
```

The table below contains WER results for various splits.

|                     |          WER |
|:-------------------:|-------------:|
| test-clean          |       0.0310 |
| test-other          |       0.0805 |
| dev-clean           |       0.0314 |
| dev-other           |       0.0827 |
