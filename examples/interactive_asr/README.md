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

Example command:
Save the dictionary, sentence piece model and model in data

python interactive_asr.py ./data --max-tokens 10000000 --nbest 1 --path ./data/model.pt --beam 40 --task speech_recognition --user-dir ../fairseq/examples/speech_recognition
