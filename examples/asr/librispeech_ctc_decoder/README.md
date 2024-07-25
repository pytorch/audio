# Speech Recognition Inference with CTC Beam Search Decoder

This is an example inference script for running decoding on the LibriSpeech dataset and wav2vec 2.0 models, using a CTC beam search decoder that supports lexicon constraint and language model integration. The language model used is a 4-gram KenLM trained on the LibriSpeech dataset.

## Usage
Additional command line parameters and information can is available with the `--help` option.

Sample command

```
python inference.py \
    --librispeech_path ./librispeech/ \
    --split test-other \
    --model WAV2VEC2_ASR_BASE_960H \
    --beam-size 1500 \
    --lm-weight 1.74 \
    --word-score 0.52
```

## Results
The table below contains WER results for various pretrained models on LibriSpeech, using a beam size of 1500, and language model weight and word insertion scores taken from Table 7 of [wav2vec 2.0](https://arxiv.org/pdf/2006.11477.pdf).

|                                                                                            Model | LM weight | word insertion | dev-clean | dev-other | test-clean | test-other |
|:------------------------------------------------------------------------------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| [WAV2VEC2_ASR_BASE_10M](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M.html#torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M)     |        3.23|        -0.26|        9.41|        15.95|        9.35|       15.91|
| [WAV2VEC2_ASR_BASE_100H](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_100H.html#torchaudio.pipelines.WAV2VEC2_ASR_BASE_100H)   |        2.15|        -0.52|        3.08|        7.89|        3.42|        8.07|
| [WAV2VEC2_ASR_BASE_960H](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html#torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H)   |        1.74|        0.52|        2.56|        6.26|        2.61|        6.15|
| [WAV2VEC2_ASR_LARGE_960H](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H.html#torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H) |        1.74|        0.52|        2.14|        4.62|        2.34|        4.98|
| [WAV2VEC2_ASR_LARGE_LV60K_10M](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_10M.html#torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_10M) |        3.86|        -1.18|        6.77|        10.03|        6.87|        10.51|
| [WAV2VEC2_ASR_LARGE_LV60K_100H](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_100H.html#torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_100H) |        2.15|        -0.52|        2.19|        4.55|        2.32|        4.64|
| [WAV2VEC2_ASR_LARGE_LV60K_960H](https://pytorch.org/audio/main/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H.html#torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H) |        1.57|        -0.64|        1.78|        3.51|        2.03|        3.68|
| [HUBERT_ASR_LARGE](https://pytorch.org/audio/main/generated/torchaudio.pipelines.HUBERT_ASR_LARGE.html#torchaudio.pipelines.HUBERT_ASR_LARGE) |        1.57|        -0.64|        1.77|        3.32|        2.03|        3.68|
| [HUBERT_ASR_XLARGE](https://pytorch.org/audio/main/generated/torchaudio.pipelines.HUBERT_ASR_XLARGE.html#torchaudio.pipelines.HUBERT_ASR_XLARGE) |        1.57|        -0.64|        1.73|        2.72|        1.90|        3.16|
