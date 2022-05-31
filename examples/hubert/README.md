# HuBERT Pre-training Example

This directory contains sample implementations of pre-training pipeline for [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447).

## Usage

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
srun --gpus-per-node=8 --ntasks-per-node=8 -N 4 --cpus-per-task=10 python train.py --root-path ./exp/data/mfcc/ --exp-dir ./exp_iter1 --feature-type mfcc --num-class 100 --max-updates 250000 --learning-rate 0.0005 --gpus 8 --num-nodes 4
```

### Pre-processing (2nd iteration)
After the first iteration of pre-training, the intermediate transformer layer's output of the pre-trained HuBERTPretrainModel can be applied to train a new KMeans clustering model. Then the KMeans clustering model can be used to generate new clustering labels for the second iteration of masked prediction training.

Sample SLURM command for the second iteration of pre-preprocessing. The 6-th transformer layer's output is used as the input feature for training KMeans model. Note that the number of clusters is increased to 500 to improve the performance.
```
srun --cpus-per-task=24 python preprocess.py --root-dir /home/datasets --feat-type hubert --exp-dir ./exp --layer-index 6 --checkpoint-path ./exp_iter1/checkpoints_librispeech_hubert_pretrain_base/xxx.ckpt --num-cluster 500
```

### Pre-training (2nd iteration)
The second iteration is trained for 400k steps.

Sample SLURM command for the second iteration of pre-training:
```
srun --gpus-per-node=8 --ntasks-per-node=8 -N 4 --cpus-per-task=10 python train.py --root-path ./exp/data/hubert_6/ --exp-dir ./exp_iter2 --feature-type hubert --num-class 500 --max-updates 400000 --learning-rate 0.0005 --gpus 8 --num-nodes 4
```
