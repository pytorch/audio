<p align="center"><img width="160" src="doc/lip_white.png" alt="logo"></p>
<h1 align="center">RNN-T ASR/VSR/AV-ASR Examples</h1>

This repository contains sample implementations of training and evaluation pipelines for RNNT based automatic, visual, and audio-visual (ASR, VSR, AV-ASR) models on LRS3. This repository includes both streaming/non-streaming modes. We follow the same training pipeline as [AutoAVSR](https://arxiv.org/abs/2303.14307).

## Preparation
1. Setup the environment.
```
conda create -y -n autoavsr python=3.8
conda activate autoavsr
```

2. Install PyTorch nightly version (Pytorch, Torchvision, Torchaudio) from [source](https://pytorch.org/get-started/), along with all necessary packages:

```Shell
pip install pytorch-lightning sentencepiece
```

3. Preprocess LRS3 to a cropped-face dataset from the [data_prep](./data_prep) folder.

4. `[sp_model_path]` is a sentencepiece model to encode targets, which can be generated using `train_spm.py`.

### Training ASR or VSR model

- `[root_dir]` is the root directory for the LRS3 cropped-face dataset.
- `[modality]` is the input modality type, including `v`, `a`, and `av`.
- `[mode]` is the model type, including `online` and `offline`.


```Shell

python train.py --root-dir [root_dir] \
                --sp-model-path ./spm_unigram_1023.model
                --exp-dir ./exp \
                --num-nodes 8 \
                --gpus 8 \
                --md [modality] \
                --mode [mode]
```

### Training AV-ASR model

```Shell
python train.py --root-dir [root-dir] \
                --sp-model-path ./spm_unigram_1023.model
                --exp-dir ./exp \
                --num-nodes 8 \
                --gpus 8 \
                --md av \
                --mode [mode]
```

### Evaluating models

```Shell
python eval.py --dataset-path [dataset_path] \
               --sp-model-path ./spm_unigram_1023.model
               --md [modality] \
               --mode [mode] \
               --checkpoint-path [checkpoint_path]
```

The table below contains WER for AV-ASR models [offline evaluation].

|    Model    |    WER [%]   |   Params (M)   |
|:-----------:|:------------:|:--------------:|
| Non-streaming models       |                |
|    AV-ASR   |      4.0     |       50       |
| Streaming models           |                |
|    AV-ASR   |      4.3     |       40       |
