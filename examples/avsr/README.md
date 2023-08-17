<p align="center"><img width="160" src="https://download.pytorch.org/torchaudio/doc-assets/avsr/lip_white.png" alt="logo"></p>
<h1 align="center">Real-time ASR/VSR/AV-ASR Examples</h1>

<div align="center">

[📘Introduction](#introduction) |
[📊Training](#Training) |
[🔮Evaluation](#Evaluation)
</div>

## Introduction

This directory contains the training recipe for real-time audio, visual, and audio-visual speech recognition (ASR, VSR, AV-ASR) models, which is an extension of [Auto-AVSR](https://arxiv.org/abs/2303.14307).

Please refer to [this tutorial](https://pytorch.org/audio/main/tutorials/device_avsr.html) for real-time AV-ASR inference from microphone and camera.

## Preparation

1. Install PyTorch (pytorch, torchvision, torchaudio) from [source](https://pytorch.org/get-started/), along with all necessary packages:

```Shell
pip install torch torchvision torchaudio pytorch-lightning sentencepiece
```

2. Preprocess LRS3. See the instructions in the [data_prep](./data_prep) folder.

## Usage

### Training

```Shell
python train.py --exp-dir=[exp_dir] \
                --exp-name=[exp_name] \
                --modality=[modality] \
                --mode=[mode] \
                --root-dir=[root-dir] \
                --sp-model-path=[sp_model_path] \
                --num-nodes=[num_nodes] \
                --gpus=[gpus]
```

- `exp-dir` and `exp-name`: The directory where the checkpoints will be saved, will be stored at the location `[exp_dir]`/`[exp_name]`.
- `modality`: Type of the input modality. Valid values are: `video`, `audio`, and `audiovisual`.
- `mode`: Type of the mode. Valid values are: `online` and `offline`.
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `sp-model-path`: Path to the sentencepiece model. Default: `./spm_unigram_1023.model`, which can be produced using `train_spm.py`.
- `num-nodes`: The number of machines used. Default: 4.
- `gpus`: The number of gpus in each machine. Default: 8.

### Evaluation

```Shell
python eval.py --modality=[modality] \
               --mode=[mode] \
               --root-dir=[dataset_path] \
               --sp-model-path=[sp_model_path] \
               --checkpoint-path=[checkpoint_path]
```

- `modality`: Type of the input modality. Valid values are: `video`, `audio`, and `audiovisual`.
- `mode`: Type of the mode. Valid values are: `online` and `offline`.
- `root-dir`: Path to the root directory where all preprocessed files will be stored.
- `sp-model-path`: Path to the sentencepiece model. Default: `./spm_unigram_1023.model`.
- `checkpoint-path`: Path to a pretraned model.

## Results

The table below contains WER for AV-ASR models that were trained from scratch [offline evaluation].

|         Model        | Training dataset (hours) | WER [%] | Params (M) |
|:--------------------:|:------------------------:|:-------:|:----------:|
| Non-streaming models |                          |         |            |
|        AV-ASR        |        LRS3 (438)        |   3.9   |     50     |
|  Streaming models    |                          |         |            |
|        AV-ASR        |        LRS3 (438)        |   3.9   |     40     |
