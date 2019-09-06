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
# Install dependencies
conda install -c pytorch torchaudio
conda install -c conda-forge librosa
conda install pyaudio
pip install sentencepiece

# Install fairseq from source
git clone https://github.com/pytorch/fairseq
cd fairseq
export CFLAGS='-stdlib=libc++'  # For Mac only
pip install --editable .
cd ..

# Install dictionary, sentence piece model, and model
wget -O ./data/dict.txt https://download.pytorch.org/models/audio/dict.txt
wget -O ./data/spm.model https://download.pytorch.org/models/audio/spm.model
wget -O ./data/model.pt https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt
```

## Run
On a file
```bash
INPUT_FILE=./data/sample.wav
python asr.py ./data --input_file $INPUT_FILE --max-tokens 10000000 --nbest 1 --path ./data/model.pt --beam 40 --task speech_recognition --user-dir ./fairseq/examples/speech_recognition
```

As a microphone
```bash
python asr.py ./data --max-tokens 10000000 --nbest 1 --path ./data/model.pt --beam 40 --task speech_recognition --user-dir ./fairseq/examples/speech_recognition
```
