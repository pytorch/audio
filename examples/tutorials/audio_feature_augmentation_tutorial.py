# -*- coding: utf-8 -*-
"""
Audio Feature Augmentation
==========================

**Author**: `Moto Hira <moto@meta.com>`__

"""

# When running this tutorial in Google Colab, install the required packages
# with the following.
# !pip install torchaudio librosa

import torch
import torchaudio
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# Preparation
# -----------
#

import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from torchaudio.utils import download_asset
import torchaudio

######################################################################
# In this tutorial, we will use a speech data from
# `VOiCES dataset <https://iqtlabs.github.io/voices/>`__,
# which is licensed under Creative Commos BY 4.0.

SAMPLE_WAV_SPEECH_PATH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")


def _get_sample(path):
    return torchaudio.load(path)

def get_speech_sample():
    return _get_sample(SAMPLE_WAV_SPEECH_PATH)


def get_spectrogram(
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    waveform, _ = get_speech_sample()
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)


######################################################################
# SpecAugment
# -----------
#
# `SpecAugment <https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html>`__
# is a popular spectrogram augmentation technique.
#
# ``torchaudio`` implements :py:func:`torchaudio.transforms.TimeStretch`,
# :py:func:`torchaudio.transforms.TimeMasking` and
# :py:func:`torchaudio.transforms.FrequencyMasking`.
#

######################################################################
# TimeStretch
# -----------
#


spec = get_spectrogram(power=None)
stretch = T.TimeStretch()

spec_12 = stretch(spec, overriding_rate=1.2)
spec_09 = stretch(spec, overriding_rate=0.9)


######################################################################
# Visualization
# ~~~~~~~~~~~~~
def plot():
    def plot_spec(ax, spec, title):
        ax.set_title(title)
        ax.imshow(librosa.amplitude_to_db(spec), origin="lower", aspect="auto")

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    plot_spec(axes[0], torch.abs(spec_12[0]), title="Stretched x1.2")
    plot_spec(axes[1], torch.abs(spec[0]), title="Original")
    plot_spec(axes[2], torch.abs(spec_09[0]), title="Stretched x0.9")
    fig.tight_layout()


plot()


######################################################################
# Audio Samples
# ~~~~~~~~~~~~~
def preview(spec, rate=16000):
    ispec = T.InverseSpectrogram()
    waveform = ispec(spec)

    return Audio(waveform[0].numpy().T, rate=rate)


preview(spec)


######################################################################
#
preview(spec_12)


######################################################################
#
preview(spec_09)


######################################################################
# Time and Frequency Masking
# --------------------------
#

torch.random.manual_seed(4)

time_masking = T.TimeMasking(time_mask_param=80)
freq_masking = T.FrequencyMasking(freq_mask_param=80)

spec = get_spectrogram()
time_masked = time_masking(spec)
freq_masked = freq_masking(spec)

######################################################################
#


def plot():
    def plot_spec(ax, spec, title):
        ax.set_title(title)
        ax.imshow(librosa.power_to_db(spec), origin="lower", aspect="auto")

    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
    plot_spec(axes[0], spec[0], title="Original")
    plot_spec(axes[1], time_masked[0], title="Masked along time axis")
    plot_spec(axes[2], freq_masked[0], title="Masked along frequency axis")
    fig.tight_layout()


plot()
