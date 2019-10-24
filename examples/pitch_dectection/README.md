# Pitch Detection Algorithm

Since the pitch is important in translating languages such as mandarin, we want a pitch detection algorithm.

We implement this using normalized cross-correlation function (NCCF) and median smoothing, mentioned in [RAPT](https://www.ee.columbia.edu/~dpwe/papers/Talkin95-rapt.pdf). Kaldi also uses NCCF, but uses an algorithm based on viterbi instead of median smoothing, see [here](https://www.danielpovey.com/files/2014_icassp_pitch.pdf).

Test by running the script. The test audio files are from [here](https://www.mediacollege.com/audio/tone/download/).
```
python pitch.py
```
