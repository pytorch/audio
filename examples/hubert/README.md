# HuBERT Pre-training and Fine-tuning Examples

This directory contains sample implementations of pre-training pipeline for [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447).

## Pre-training Usage

The Base architecture of HuBERT model requires two iterations of pre-training.
### Pre-processing (1st iteration)
[`preprocess.py`](./preprocess.py) generates the file list of training and validation data, trains a KMeans clustering model with either MFCC feature or the transformer layer's output from the pre-trained HuBERT model, then predict the cluster ID for each utterance as the label for masked prediction training.

Sample SLURM command for the first iteration of pre-preprocessing, which uses MFCC feature to train KMeans model:
```
srun --cpus-per-task=24 python preprocess.py --root-dir /home/datasets --feat-type mfcc --exp-dir ./exp --num-cluster 100
```

### Pre-training (1st iteration)

[`train.py`](./train.py) trains a HuBERTPretrainModel using PyTorch Lightning. Note that the script expects users to have access to GPU nodes for training.

The first iteration is trained for 250k steps on 32 GPUs, each GPU has at most 87.5 seconds of audio in a mini-batch.

Sample SLURM command for the first iteration of pre-training:
```
srun --gpus-per-node=8 --ntasks-per-node=8 -N 4 --cpus-per-task=10 python train.py --dataset-path ./exp/data/mfcc/ --exp-dir ./exp_iter1 --feature-type mfcc --num-class 100 --max-updates 250000 --learning-rate 0.0005 --gpus 8 --num-nodes 4
```

### Pre-processing (2nd iteration)
After the first iteration of pre-training, the intermediate transformer layer's output of the pre-trained HuBERTPretrainModel can be applied to train a new KMeans clustering model. Then the KMeans clustering model can be used to generate new clustering labels for the second iteration of masked prediction training.

Sample SLURM command for the second iteration of pre-preprocessing. The 6-th transformer layer's output is used as the input feature for training KMeans model. Note that the number of clusters is increased to 500 to improve the performance.
```
srun --cpus-per-task=24 python preprocess.py --root-dir /home/datasets --feat-type hubert --exp-dir ./exp --layer-index 6 --checkpoint-path ./exp_iter1/ --num-rank 40 checkpoints_librispeech_hubert_pretrain_base/xxx.ckpt --num-cluster 500 --percent 0.1
```

### Pre-training (2nd iteration)
The second iteration is trained for 400k steps.

Sample SLURM command for the second iteration of pre-training:
```
srun --gpus-per-node=8 --ntasks-per-node=8 -N 4 --cpus-per-task=10 python train.py --dataset-path ./exp/data/hubert_6/ --exp-dir ./exp_iter2 --feature-type hubert --num-class 500 --max-updates 400000 --learning-rate 0.0005 --gpus 8 --num-nodes 4
```

## Fine-tuning Usage

After finishing the pre-training step, the model can be validated by fine-tuning on the `LibriLightLimited` dataset (the supervised subset of [Libri-Light](https://github.com/facebookresearch/libri-light) dataset) with an extra feed-forward layer on top of the transformer layers.

During the whole fine-tuning process, the feature extraction layers are frozen (i.e., no gradients is back propagated to these layers). For the first 10k fine-tuning iterations, the transformer layers are frozen and only the CTC layer is trained. After 10k iterations, the transformer layers are fine-tuned along with the CTC layer.

Sample SLURM command for fine-tuning on `10h` subset of `LibriLightLimited` dataset:
```
srun --gpus-per-node=1 -N 1 --ntasks-per-node=1 --cpus-per-task=10 \
  python finetune.py --dataset-path /root/datasets/ --exp-dir ./exp_finetune \
  --checkpoint ./exp_iter2/checkpoints_librispeech_hubert_pretrain_base/epoch=361-step=399999.ckpt \
  --gpus 1 --debug --warmup-updates 2000 --hold-updates 8000 --decay-updates 10000 --max-updates 20000 --learning-rate 5e-5
```

# Decoding

### Viterbi Decoding
The output of CTC layer contains repeated letters, blank symbol ("-"), and silence symbol ("|"). Viterbi decoding unifies the repeated letters into a single letter, removes the blank symbol, and splits the string into a list of words by the silence symbol.

Sample SLURM command for evaluation with Viterbi decoding:
```
srun python evaluate.py --librispeech_path /root/datasets/ --checkpoint ./exp_finetune/checkpoints_hubert_pretrain_base/epoch\=109-step\=19999.ckpt --split test-clean
```

### CTC Decoding with language model
torchaudio provides a CTCDecoder feature that is based on [Flashlight](https://github.com/flashlight/flashlight). The decoder supports KenLM language model. Use `--use-lm` to enable CTC decoding with KenLM 4-gram language model.

Sample SLURM command for evaluation with KenLM language model (use the checkpoint that has the lowest validation loss):
```
srun python evaluate.py --librispeech_path /root/datasets/ --checkpoint ./exp_finetune/checkpoints_hubert_pretrain_base/epoch\=106-step\=19500.ckpt --split test-clean --use-lm --beam-size 1500 --lm-weight 2.46 --word-score -0.59
```

### WER results
The table below contains WER results for fine-tuning HuBERT Base model on `10h` subset of `LibriLightLimited` dataset.

|                   | WER% (Viterbi)|  WER% (KenLM) |
|:-----------------:|--------------:|--------------:|
| dev-clean         |       10.9    |       4.2     |
| dev-other         |       17.5    |       9.4     |
| test-clean        |       10.9    |       4.4     |
| test-other        |       17.8    |       9.5     |
