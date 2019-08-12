# asr-demo

We recommend that you use [conda](https://docs.conda.io/en/latest/miniconda.html) to install these dependencies.

What you need to run this demo
- [python3](https://www.python.org/download/releases/3.0/)
- [torchaudio](https://github.com/pytorch/audio/tree/master/torchaudio)
- [pytorch](https://pytorch.org/)
- [librosa](https://librosa.github.io/librosa/)
- [fairseq](https://github.com/pytorch/fairseq) (clone the github repository)


Models:
- [dictionary](https://download.pytorch.org/models/audio/dict.txt)
- [sentence piece model](https://download.pytorch.org/models/audio/spm.model)
- [model](https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt)

Save the dictionary, sentence piece model and model in data

## Installation
```bash
# get asr-demo
git clone https://github.com/cpuhrsch/asr-demo
cd asr-demo

# install dependencies
conda install -c pytorch torchaudio
conda install -c conda-forge librosa
pip install sentencepiece

# install fairseq (must build from source)
git clone https://github.com/pytorch/fairseq
cd fairseq
export CFLAGS='-stdlib=libc++' 
pip install --editable .
cd ..

# install dictionary, sentence piece model, and model
wget -O ./data/dict.txt https://download.pytorch.org/models/audio/dict.txt
wget -O ./data/spm.model https://download.pytorch.org/models/audio/spm.model
wget -O ./data/model.pt https://download.pytorch.org/models/audio/checkpoint_avg_60_80.pt
```

## Run
On a file
```bash
INPUT_FILE=./data/sample.wav
python infer_file.py ./data --input_file $INPUT_FILE --max-tokens 10000000 --nbest 1 --path ./data/model.pt --beam 40 --task speech_recognition --user-dir ./fairseq/examples/speech_recognition
```

As a microphone
```bash
python interactive_asr.py ./data --max-tokens 10000000 --nbest 1 --path ./data/model.pt --beam 40 --task speech_recognition --user-dir ./fairseq/examples/speech_recognition
```
