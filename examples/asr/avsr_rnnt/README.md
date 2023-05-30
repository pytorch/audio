<p align="center"><img width="160" src="doc/lip_white.png" alt="logo"></p>
<h1 align="center">RNN-T ASR/VSR/AV-ASR Examples</h1>

This repository  contains sample implementations of training and evaluation pipelines for RNNT based automatic, visual, and audio-visual (ASR, VSR, AV-ASR) models on LRS3. This repository includes both streaming/non-streaming modes.

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

4. For better performance, you can choose to download [models](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo) from Table 1 to initialise ASR/VSR front-end.

### Training A/V-ASR model

- `[dataset_path]` is the directory for original dataset.
- `[label_path]` is the labels directory.
- `[modality]` is the input modality type, including `v`, `a`, and `av`.
- `[mode]` is the model type, including `online` and `offline`.

```Shell

python train.py --dataset-path [dataset_path] \
                --label-path [label-path]
                --pretrained-model-path [pretrained_model_path] \
                --sp-model-path ./spm_unigram_1023.model
                --exp-dir ./exp \
                --num-nodes 8 \
                --gpus 8 \
                --md [modality] \
                --mode [mode]
```

### Training AV-ASR model

```Shell
python train.py --dataset-path [dataset_path] \
                --label-path [label-path] 
                --pretrained-vid-model-path [pretrained_vid_model_path] \
                --pretrained-aud-model-path [pretrained_aud_model_path] \
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
               --label-path [label-path]
               --pretrained-model-path [pretrained_model_path] \
               --sp-model-path ./spm_unigram_1023.model
               --md [modality] \
               --mode [mode] \
               --checkpoint-path [checkpoint_path]
```

The table below contains WER for AV-ASR models.

|    Model    |    WER [%]   |   Params (M)   |
|:-----------:|:------------:|:--------------:|
| Non-streaming models       |                |
|    AV-ASR   |      4.2     |       50       |
| Streaming models           |                |
|    AV-ASR   |      4.9     |       40       |
