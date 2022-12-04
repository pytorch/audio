# Contextual Conformer RNN-T with TCPGen Example

This directory contains sample implementations of training and evaluation pipelines for the Conformer RNN-T model with tree-constrained pointer generator (TCPGen) for contextual biasing.

## Setup
### Install PyTorch and TorchAudio nightly or from source
Because Conformer RNN-T is currently a prototype feature, you will need to either use the TorchAudio nightly build or build TorchAudio from source. Note also that GPU support is required for training.

To install the nightly, follow the directions at <https://pytorch.org/>.

To build TorchAudio from source, refer to the [contributing guidelines](https://github.com/pytorch/audio/blob/main/CONTRIBUTING.md).

### Install additional dependencies
```bash
pip install pytorch-lightning sentencepiece
```

## Usage

### Training

[`train.py`](./train.py) trains an Conformer RNN-T model (30.2M parameters, 121MB) on LibriSpeech using PyTorch Lightning. Note that the script expects users to have the following:
- Access to GPU nodes for training.
- Full LibriSpeech dataset.
- SentencePiece model to be used to encode targets; the model can be generated using [`train_spm.py`](./train_spm.py).
- File (--global_stats_path) that contains training set feature statistics; this file can be generated using [`global_stats.py`](../emformer_rnnt/global_stats.py).

Sample local training script: `train.sh`

Training options:

### Evaluation

[`eval.py`](./eval.py) evaluates a trained Conformer RNN-T model on LibriSpeech test-clean.

Sample decoding script: `eval.sh`

Decoding options:

