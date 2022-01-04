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
