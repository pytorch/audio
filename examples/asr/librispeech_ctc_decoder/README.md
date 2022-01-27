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

|                                                                                            Model | test-clean | test-other |
|:------------------------------------------------------------------------------------------------:|-----------:|-----------:|
| [WAV2VEC2_ASR_BASE_10M](https://pytorch.org/audio/main/pipelines.html#wav2vec2-asr-base-10m)     |        9.35|       15.91|
| [WAV2VEC2_ASR_BASE_100H](https://pytorch.org/audio/main/pipelines.html#wav2vec2-asr-base-100h)   |        3.42|        8.07|
| [WAV2VEC2_ASR_BASE_960H](https://pytorch.org/audio/main/pipelines.html#wav2vec2-asr-base-960h)   |        2.61|        6.15|
| [WAV2VEC2_ASR_LARGE_960H](https://pytorch.org/audio/main/pipelines.html#wav2vec2-asr-large-960h) |        2.34|        4.98|
