import os
import sys
import torchaudio

if __name__ == '__main__':

    AUDIO_PATH = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    N_FFT = 512
    THRESHOLD_HZ = 3000

    audio, sample_rate = torchaudio.load(AUDIO_PATH, normalization=False)
    lowpassed_audio = torchaudio.functional.lowpass(audio, sample_rate, N_FFT, THRESHOLD_HZ)
    torchaudio.save(OUTPUT_PATH, lowpassed_audio, sample_rate)