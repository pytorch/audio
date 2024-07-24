"""
Audio Datasets
==============

**Author**: `Moto Hira <moto@meta.com>`__

``torchaudio`` provides easy access to common, publicly accessible
datasets. Please refer to the official documentation for the list of
available datasets.
"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
#

import os

import IPython

import matplotlib.pyplot as plt


_SAMPLE_DIR = "_assets"
YESNO_DATASET_PATH = os.path.join(_SAMPLE_DIR, "yes_no")
os.makedirs(YESNO_DATASET_PATH, exist_ok=True)


def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()


######################################################################
# Here, we show how to use the
# :py:class:`torchaudio.datasets.YESNO` dataset.
#

dataset = torchaudio.datasets.YESNO(YESNO_DATASET_PATH, download=True)

######################################################################
#
i = 1
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#
i = 3
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#
i = 5
waveform, sample_rate, label = dataset[i]
plot_specgram(waveform, sample_rate, title=f"Sample {i}: {label}")
IPython.display.Audio(waveform, rate=sample_rate)
