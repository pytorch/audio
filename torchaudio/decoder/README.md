# Flashlight Decoder Binding
CTC Decoder with KenLM and lexicon support based on [flashlight](https://github.com/flashlight/flashlight) decoder implementation and fairseq [KenLMDecoder](https://github.com/pytorch/fairseq/blob/fcca32258c8e8bcc9f9890bf4714fa2f96b6b3e1/examples/speech_recognition/new/decoders/flashlight_decoder.py#L53) Python wrapper

## Setup
### Build KenLM
- Install KenLM following the instructions [here](https://github.com/kpu/kenlm#compiling) 
- set `KENLM_ROOT` variable to the KenLM installation path
### Build torchaudio with decoder support
```
BUILD_FL_DECODER=1 USE_KENLM=1 python setup.py develop
```

## Usage
```py
from torchaudio.decoder import kenlm_ctc_lexicon_decoder
decoder = kenlm_ctc_lexicon_decoder(args...)
results = decoder(emissions) # dim (B, nbest) of dictionary of "tokens", "score", "words" keys
best_transcript = " ".join(results[0][0]["words"]).strip()
```

## Required Files
- tokens: tokens for which the acoustic model generates probabilities for
- lexicon: mapping between words and its corresponding spelling
- language model: n-gram KenLM model