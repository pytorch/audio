# -*- coding: utf-8 -*-
"""
Audio Data Augmentation
=======================

``torchaudio`` provides a variety of ways to augment audio data.
"""

import torch
import torchaudio
import torchaudio.functional as F

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# Preparing data and utility functions (skip this section)
# --------------------------------------------------------
#

# @title Prepare data and utility functions. {display-mode: "form"}
# @markdown
# @markdown You do not need to look into this cell.
# @markdown Just execute once and you are good to go.
# @markdown
# @markdown In this tutorial, we will use a speech data from [VOiCES dataset](https://iqtlabs.github.io/voices/),
# @markdown which is licensed under Creative Commos BY 4.0.

# -------------------------------------------------------------------------------
# Preparation of data and helper functions.
# -------------------------------------------------------------------------------

import math
import os

import matplotlib.pyplot as plt
import requests
from IPython.display import Audio, display


_SAMPLE_DIR = "_assets"

SAMPLE_WAV_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/steam-train-whistle-daniel_simon.wav"
SAMPLE_WAV_PATH = os.path.join(_SAMPLE_DIR, "steam.wav")

SAMPLE_RIR_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/room-response/rm1/impulse/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav"  # noqa: E501
SAMPLE_RIR_PATH = os.path.join(_SAMPLE_DIR, "rir.wav")

SAMPLE_WAV_SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"  # noqa: E501
SAMPLE_WAV_SPEECH_PATH = os.path.join(_SAMPLE_DIR, "speech.wav")

SAMPLE_NOISE_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/distant-16k/distractors/rm1/babb/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav"  # noqa: E501
SAMPLE_NOISE_PATH = os.path.join(_SAMPLE_DIR, "bg.wav")

os.makedirs(_SAMPLE_DIR, exist_ok=True)


def _fetch_data():
    uri = [
        (SAMPLE_WAV_URL, SAMPLE_WAV_PATH),
        (SAMPLE_RIR_URL, SAMPLE_RIR_PATH),
        (SAMPLE_WAV_SPEECH_URL, SAMPLE_WAV_SPEECH_PATH),
        (SAMPLE_NOISE_URL, SAMPLE_NOISE_PATH),
    ]
    for url, path in uri:
        with open(path, "wb") as file_:
            file_.write(requests.get(url).content)


_fetch_data()


def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.sox_effects.apply_effects_file(path, effects=effects)


def get_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_PATH, resample=resample)


def get_speech_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        return Audio(waveform[0], rate=sample_rate)
    elif num_channels == 2:
        return Audio((waveform[0], waveform[1]), rate=sample_rate)
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def get_rir_sample(*, resample=None, processed=False):
    rir_raw, sample_rate = _get_sample(SAMPLE_RIR_PATH, resample=resample)
    if not processed:
        return rir_raw, sample_rate
    rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
    rir = rir / torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    return rir, sample_rate


def get_noise_sample(*, resample=None):
    return _get_sample(SAMPLE_NOISE_PATH, resample=resample)


######################################################################
# Applying effects and filtering
# ------------------------------
#
# :py:func:`torchaudio.sox_effects` allows for directly applying filters similar to
# those available in ``sox`` to Tensor objects and file object audio sources.
#
# There are two functions for this:
#
# -  :py:func:`torchaudio.sox_effects.apply_effects_tensor` for applying effects
#    to Tensor.
# -  :py:func:`torchaudio.sox_effects.apply_effects_file` for applying effects to
#    other audio sources.
#
# Both functions accept effect definitions in the form
# ``List[List[str]]``.
# This is mostly consistent with how ``sox`` command works, but one caveat is
# that ``sox`` adds some effects automatically, whereas ``torchaudio``’s
# implementation does not.
#
# For the list of available effects, please refer to `the sox
# documentation <http://sox.sourceforge.net/sox.html>`__.
#
# **Tip** If you need to load and resample your audio data on the fly,
# then you can use :py:func:`torchaudio.sox_effects.apply_effects_file`
# with effect ``"rate"``.
#
# **Note** :py:func:`torchaudio.sox_effects.apply_effects_file` accepts a
# file-like object or path-like object.
# Similar to :py:func:`torchaudio.load`, when the audio format cannot be
# inferred from either the file extension or header, you can provide
# argument ``format`` to specify the format of the audio source.
#
# **Note** This process is not differentiable.
#


# Load the data
waveform1, sample_rate1 = get_sample(resample=16000)

# Define effects
effects = [
    ["lowpass", "-1", "300"],  # apply single-pole lowpass filter
    ["speed", "0.8"],  # reduce the speed
    # This only changes sample rate, so it is necessary to
    # add `rate` effect with original sample rate after this.
    ["rate", f"{sample_rate1}"],
    ["reverb", "-w"],  # Reverbration gives some dramatic feeling
]

# Apply effects
waveform2, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(waveform1, sample_rate1, effects)

print_stats(waveform1, sample_rate=sample_rate1, src="Original")
print_stats(waveform2, sample_rate=sample_rate2, src="Effects Applied")

######################################################################
# Note that the number of frames and number of channels are different from
# those of the original after the effects are applied. Let’s listen to the
# audio.
#

######################################################################
# Original:
# ~~~~~~~~~
#

plot_waveform(waveform1, sample_rate1, title="Original", xlim=(-0.1, 3.2))
plot_specgram(waveform1, sample_rate1, title="Original", xlim=(0, 3.04))
play_audio(waveform1, sample_rate1)

######################################################################
# Effects applied:
# ~~~~~~~~~~~~~~~~
#

plot_waveform(waveform2, sample_rate2, title="Effects Applied", xlim=(-0.1, 3.2))
plot_specgram(waveform2, sample_rate2, title="Effects Applied", xlim=(0, 3.04))
play_audio(waveform2, sample_rate2)

######################################################################
# Doesn’t it sound more dramatic?
#

######################################################################
# Simulating room reverberation
# -----------------------------
#
# `Convolution
# reverb <https://en.wikipedia.org/wiki/Convolution_reverb>`__ is a
# technique that's used to make clean audio sound as though it has been
# produced in a different environment.
#
# Using Room Impulse Response (RIR), for instance, we can make clean speech
# sound as though it has been uttered in a conference room.
#
# For this process, we need RIR data. The following data are from the VOiCES
# dataset, but you can record your own — just turn on your microphone
# and clap your hands.
#


sample_rate = 8000

rir_raw, _ = get_rir_sample(resample=sample_rate)

plot_waveform(rir_raw, sample_rate, title="Room Impulse Response (raw)", ylim=None)
plot_specgram(rir_raw, sample_rate, title="Room Impulse Response (raw)")
play_audio(rir_raw, sample_rate)

######################################################################
# First, we need to clean up the RIR. We extract the main impulse, normalize
# the signal power, then flip along the time axis.
#

rir = rir_raw[:, int(sample_rate * 1.01) : int(sample_rate * 1.3)]
rir = rir / torch.norm(rir, p=2)
rir = torch.flip(rir, [1])

print_stats(rir)
plot_waveform(rir, sample_rate, title="Room Impulse Response", ylim=None)

######################################################################
# Then, we convolve the speech signal with the RIR filter.
#

speech, _ = get_speech_sample(resample=sample_rate)

speech_ = torch.nn.functional.pad(speech, (rir.shape[1] - 1, 0))
augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

######################################################################
# Original:
# ~~~~~~~~~
#

plot_waveform(speech, sample_rate, title="Original", ylim=None)
plot_specgram(speech, sample_rate, title="Original")
play_audio(speech, sample_rate)

######################################################################
# RIR applied:
# ~~~~~~~~~~~~
#

plot_waveform(augmented, sample_rate, title="RIR Applied", ylim=None)
plot_specgram(augmented, sample_rate, title="RIR Applied")
play_audio(augmented, sample_rate)


######################################################################
# Adding background noise
# -----------------------
#
# To add background noise to audio data, you can simply add a noise Tensor to
# the Tensor representing the audio data. A common method to adjust the
# intensity of noise is changing the Signal-to-Noise Ratio (SNR).
# [`wikipedia <https://en.wikipedia.org/wiki/Signal-to-noise_ratio>`__]
#
# $$ \\mathrm{SNR} = \\frac{P_{signal}}{P_{noise}} $$
#
# $$ \\mathrm{SNR_{dB}} = 10 \\log _{{10}} \\mathrm {SNR} $$
#


sample_rate = 8000
speech, _ = get_speech_sample(resample=sample_rate)
noise, _ = get_noise_sample(resample=sample_rate)
noise = noise[:, : speech.shape[1]]

speech_power = speech.norm(p=2)
noise_power = noise.norm(p=2)

snr_dbs = [20, 10, 3]
noisy_speeches = []
for snr_db in snr_dbs:
    snr = math.exp(snr_db / 10)
    scale = snr * noise_power / speech_power
    noisy_speeches.append((scale * speech + noise) / 2)

######################################################################
# Background noise:
# ~~~~~~~~~~~~~~~~~
#

plot_waveform(noise, sample_rate, title="Background noise")
plot_specgram(noise, sample_rate, title="Background noise")
play_audio(noise, sample_rate)

######################################################################
# SNR 20 dB:
# ~~~~~~~~~~
#

snr_db, noisy_speech = snr_dbs[0], noisy_speeches[0]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
play_audio(noisy_speech, sample_rate)

######################################################################
# SNR 10 dB:
# ~~~~~~~~~~
#

snr_db, noisy_speech = snr_dbs[1], noisy_speeches[1]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
play_audio(noisy_speech, sample_rate)

######################################################################
# SNR 3 dB:
# ~~~~~~~~~~
#

snr_db, noisy_speech = snr_dbs[2], noisy_speeches[2]
plot_waveform(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
plot_specgram(noisy_speech, sample_rate, title=f"SNR: {snr_db} [dB]")
play_audio(noisy_speech, sample_rate)

######################################################################
# Applying codec to Tensor object
# -------------------------------
#
# :py:func:`torchaudio.functional.apply_codec` can apply codecs to
# a Tensor object.
#
# **Note** This process is not differentiable.
#


waveform, sample_rate = get_speech_sample(resample=8000)

plot_specgram(waveform, sample_rate, title="Original")

configs = [
    ({"format": "wav", "encoding": "ULAW", "bits_per_sample": 8}, "8 bit mu-law"),
    ({"format": "gsm"}, "GSM-FR"),
    ({"format": "mp3", "compression": -9}, "MP3"),
    ({"format": "vorbis", "compression": -1}, "Vorbis"),
]
waveforms = []
for param, title in configs:
    augmented = F.apply_codec(waveform, sample_rate, **param)
    plot_specgram(augmented, sample_rate, title=title)
    waveforms.append(augmented)

######################################################################
# Original:
# ~~~~~~~~~
#

play_audio(waveform, sample_rate)

######################################################################
# 8 bit mu-law:
# ~~~~~~~~~~~~~
#

play_audio(waveforms[0], sample_rate)

######################################################################
# GSM-FR:
# ~~~~~~~
#

play_audio(waveforms[1], sample_rate)

######################################################################
# MP3:
# ~~~~
#

play_audio(waveforms[2], sample_rate)

######################################################################
# Vorbis:
# ~~~~~~~
#

play_audio(waveforms[3], sample_rate)

######################################################################
# Simulating a phone recoding
# ---------------------------
#
# Combining the previous techniques, we can simulate audio that sounds
# like a person talking over a phone in a echoey room with people talking
# in the background.
#

sample_rate = 16000
original_speech, _ = get_speech_sample(resample=sample_rate)

plot_specgram(original_speech, sample_rate, title="Original")

# Apply RIR
rir, _ = get_rir_sample(resample=sample_rate, processed=True)
speech_ = torch.nn.functional.pad(original_speech, (rir.shape[1] - 1, 0))
rir_applied = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

plot_specgram(rir_applied, sample_rate, title="RIR Applied")

# Add background noise
# Because the noise is recorded in the actual environment, we consider that
# the noise contains the acoustic feature of the environment. Therefore, we add
# the noise after RIR application.
noise, _ = get_noise_sample(resample=sample_rate)
noise = noise[:, : rir_applied.shape[1]]

snr_db = 8
scale = math.exp(snr_db / 10) * noise.norm(p=2) / rir_applied.norm(p=2)
bg_added = (scale * rir_applied + noise) / 2

plot_specgram(bg_added, sample_rate, title="BG noise added")

# Apply filtering and change sample rate
filtered, sample_rate2 = torchaudio.sox_effects.apply_effects_tensor(
    bg_added,
    sample_rate,
    effects=[
        ["lowpass", "4000"],
        [
            "compand",
            "0.02,0.05",
            "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
            "-8",
            "-7",
            "0.05",
        ],
        ["rate", "8000"],
    ],
)

plot_specgram(filtered, sample_rate2, title="Filtered")

# Apply telephony codec
codec_applied = F.apply_codec(filtered, sample_rate2, format="gsm")

plot_specgram(codec_applied, sample_rate2, title="GSM Codec Applied")


######################################################################
# Original speech:
# ~~~~~~~~~~~~~~~~
#

play_audio(original_speech, sample_rate)

######################################################################
# RIR applied:
# ~~~~~~~~~~~~
#

play_audio(rir_applied, sample_rate)

######################################################################
# Background noise added:
# ~~~~~~~~~~~~~~~~~~~~~~~
#

play_audio(bg_added, sample_rate)

######################################################################
# Filtered:
# ~~~~~~~~~
#

play_audio(filtered, sample_rate2)

######################################################################
# Codec aplied:
# ~~~~~~~~~~~~~
#

play_audio(codec_applied, sample_rate2)
