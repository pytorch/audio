# Flashlight Decoder Binding
CTC Decoder with KenLM and lexicon support based on [flashlight](https://github.com/flashlight/flashlight) decoder implementation
and fairseq [KenLMDecoder](https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/speech_recognition/new/decoders/flashlight_decoder.py#L53)
Python wrapper

## Setup
### Build torchaudio with decoder support
```
BUILD_CTC_DECODER=1 python setup.py develop
```

## Usage
```py
from torchaudio.prototype.ctc_decoder import kenlm_lexicon_decoder
decoder = kenlm_lexicon_decoder(args...)
results = decoder(emissions) # dim (B, nbest) of dictionary of "tokens", "score", "words" keys
best_transcripts = [" ".join(results[i][0].words).strip() for i in range(B)]
```

## Required Files
- tokens: tokens for which the acoustic model generates probabilities for
- lexicon: mapping between words and its corresponding spelling
- language model: n-gram KenLM model

## Experiment Results
LibriSpeech dev-other and test-other results using pretrained [Wav2Vec2](https://arxiv.org/pdf/2006.11477.pdf) models of
BASE configuration.

| Model       | Decoder    | dev-other   | test-other | beam search params                          |
| ----------- | ---------- | ----------- | ---------- |-------------------------------------------- |
| BASE_10M    | Greedy     | 51.6        | 51         |                                             |
|             | 4-gram LM  | 15.95       | 15.9       | LM weight=3.23, word score=-0.26, beam=1500 |
| BASE_100H   | Greedy     | 13.6        | 13.3       |                                             |
|             | 4-gram LM  | 8.5         | 8.8        | LM weight=2.15, word score=-0.52, beam=50   |
| BASE_960H   | Greedy     | 8.9         | 8.4        |                                             |
|             | 4-gram LM  | 6.3         | 6.4        | LM weight=1.74, word score=0.52, beam=50    |
