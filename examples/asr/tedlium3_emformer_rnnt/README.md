# Emformer RNN-T ASR Example for TED-LIUM release 3 dataset

This directory contains sample implementations of training and evaluation pipelines for an on-device-oriented streaming-capable Emformer RNN-T ASR model.

## Usage

### Training

[`train.py`](./train.py) trains an Emformer RNN-T model on TED-LIUM release 3 using PyTorch Lightning. Note that the script expects users to have access to GPU nodes for training and provide paths to the full TED-LIUM release 3 dataset and the SentencePiece model to be used to encode targets.

Sample SLURM command:
```
srun --cpus-per-task=12 --gpus-per-node=8 -N 1 --ntasks-per-node=8 python train.py --exp-dir ./experiments --tedlium-path ./datasets/ --global-stats-path ./global_stats.json --sp-model-path ./spm_bpe_500.model
```

### Evaluation

[`eval.py`](./eval.py) evaluates a trained Emformer RNN-T model on TED-LIUM release 3 test set.

The table below contains WER results for dev and test subsets of TED-LIUM release 3.

|             |          WER |
|:-----------:|-------------:|
| dev         |       0.108  |
| test        |       0.098  |


Sample SLURM command:
```
srun python eval.py --checkpoint-path ./experiments/checkpoints/epoch=119-step=254999.ckpt  --tedlium-path ./datasets/ --sp-model-path ./spm-bpe-500.model --use-cuda
```
