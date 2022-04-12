# Conformer RNN-T ASR Example

This directory contains sample implementations of training and evaluation pipelines for a Conformer RNN-T ASR model.

## Usage

### Training

[`train.py`](./train.py) trains an Conformer RNN-T model (30.2M parameters, 121MB) on LibriSpeech using PyTorch Lightning. Note that the script expects users to have access to GPU nodes for training and provide paths to the full LibriSpeech dataset and the SentencePiece model to be used to encode targets.

Sample SLURM command:
```
srun --cpus-per-task=12 --gpus-per-node=8 -N 4 --ntasks-per-node=8 python train.py --exp_dir ./experiments --librispeech_path ./librispeech/ --global_stats_path ./global_stats.json --sp_model_path ./spm_unigram_1023.model --epochs 160
```

### Evaluation

[`eval.py`](./eval.py) evaluates a trained Conformer RNN-T model on LibriSpeech test-clean.

Sample SLURM command:
```
srun python eval.py --checkpoint_path ./experiments/checkpoints/epoch=159.ckpt --librispeech_path ./librispeech/ --sp_model_path ./spm_unigram_1023.model --use_cuda
```

The table below contains WER results for various splits.

|                     |          WER |
|:-------------------:|-------------:|
| test-clean          |       0.0310 |
| test-other          |       0.0805 |
| dev-clean           |       0.0314 |
| dev-other           |       0.0827 |
