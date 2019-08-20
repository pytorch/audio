# asr-demo

To run this demo, you need the following libraries
- [python3](https://www.python.org/download/releases/3.0/)
- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [torchaudio](https://github.com/pytorch/audio/tree/master/torchaudio)
- [pytorch](https://pytorch.org/)
- [librosa](https://librosa.github.io/librosa/)
- [fairseq](https://github.com/pytorch/fairseq) (clone the github repository)
and the following models
- [dictionary](https://download.pytorch.org/models/audio/dict.txt)
- [sentence piece model](https://download.pytorch.org/models/audio/spm.model)
- [model](https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt)

## Installation

We recommend that you use [conda](https://docs.conda.io/en/latest/miniconda.html) to install the dependencies when available.
```bash
# Assume that all commands are from the examples folder
cd examples

# Install dependencies
conda install -c pytorch torchaudio
conda install -c conda-forge librosa
conda install pyaudio
pip install sentencepiece

# Install fairseq from source
git clone https://github.com/pytorch/fairseq interactive_asr/fairseq
pushd interactive_asr/fairseq
export CFLAGS='-stdlib=libc++'  # For Mac only
pip install --editable .
popd

# Install dictionary, sentence piece model, and model
wget -O interactive_asr/data/dict.txt https://download.pytorch.org/models/audio/dict.txt
wget -O interactive_asr/data/spm.model https://download.pytorch.org/models/audio/spm.model
wget -O interactive_asr/data/model.pt https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt
```

## Run
On a file
```bash
INPUT_FILE=interactive_asr/data/sample.wav
python -m interactive_asr.asr interactive_asr/data --input_file $INPUT_FILE --max-tokens 10000000 --nbest 1 \
  --path interactive_asr/data/model.pt --beam 40 --task speech_recognition \
  --user-dir interactive_asr/fairseq/examples/speech_recognition
```

As a microphone
```bash
python -m interactive_asr.asr interactive_asr/data --max-tokens 10000000 --nbest 1 \
  --path interactive_asr/data/model.pt --beam 40 --task speech_recognition \
  --user-dir interactive_asr/fairseq/examples/speech_recognition
```
To run the testcase associated with this example
```bash
ASR_MODEL_PATH=interactive_asr/data/model.pt \
ASR_INPUT_FILE=interactive_asr/data/sample.wav \
ASR_DATA_PATH=interactive_asr/data \
ASR_USER_DIR=interactive_asr/fairseq/examples/speech_recognition \
python -m unittest test/test_interactive_asr.py
```
