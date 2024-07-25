# Modularized Self-supervised Learning Recipe

This directory contains the modularized training recipe for audio/speech self-supervised learning. The principle is to let users easily inject a new component (model, data_module, loss function, etc) to the existing recipe for different tasks (e.g. Wav2Vec 2.0, HuBERT, etc).


## HuBERT Pre-training Example
To get the K-Means labels for HuBERT pre-training, please check the [pre-processing step](../hubert/README.md#pre-processing-1st-iteration) in hubert example.

In order to run the HuBERT pre-training script for the first iteration, users need to go to `examples` directory and run the following SLURM command:
```
cd examples

srun \
--gpus-per-node=8 \
--ntasks-per-node=8 \
-N 4 \
--cpus-per-task=10 \
python -m self_supervised_learning.train_hubert \
--dataset-path hubert/exp/data/mfcc/ \
--exp-dir self_supervised_learning/exp_iter1 \
--feature-type mfcc \
--num-class 100 \
--max-updates 250000 \
--learning-rate 0.0005 \
--gpus 8 \
--num-nodes 4
```
