# -*- coding: utf-8 -*-
"""
Audio I/O
=========

**Author**: `Moto Hira <moto@meta.com>`__

This tutorial shows how to use TorchAudio's basic I/O API to inspect audio data,
load them into PyTorch Tensors and save PyTorch Tensors.

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:

    - The APIs described in this tutorial are deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.

"""

import torch
import torchaudio
from torchaudio.utils import load_torchcodec
from io import BytesIO

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
# Preparation
# -----------
#
# First, we import the modules and download the audio assets we use in this tutorial.
#
# .. note::
#    When running this tutorial in Google Colab, install the required packages
#    with the following:
#
#    .. code::
#
#       !pip install boto3

import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")


def _hide_seek(obj):
    class _wrapper:
        def __init__(self, obj):
            self.obj = obj

        def read(self, n):
            return self.obj.read(n)

    return _wrapper(obj)


######################################################################
# Querying audio metadata
# -----------------------
#
# Function :py:func:`torchaudio.info` fetches audio metadata.
# You can provide a path-like object or file-like object.
#

metadata = torchaudio.info(SAMPLE_WAV)
print(metadata)

######################################################################
# Where
#
# -  ``sample_rate`` is the sampling rate of the audio
# -  ``num_channels`` is the number of channels
# -  ``num_frames`` is the number of frames per channel
# -  ``bits_per_sample`` is bit depth
# -  ``encoding`` is the sample coding format
#
# ``encoding`` can take on one of the following values:
#
# -  ``"PCM_S"``: Signed integer linear PCM
# -  ``"PCM_U"``: Unsigned integer linear PCM
# -  ``"PCM_F"``: Floating point linear PCM
# -  ``"FLAC"``: Flac, `Free Lossless Audio
#    Codec <https://xiph.org/flac/>`__
# -  ``"ULAW"``: Mu-law,
#    [`wikipedia <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`__]
# -  ``"ALAW"``: A-law
#    [`wikipedia <https://en.wikipedia.org/wiki/A-law_algorithm>`__]
# -  ``"MP3"`` : MP3, MPEG-1 Audio Layer III
# -  ``"VORBIS"``: OGG Vorbis [`xiph.org <https://xiph.org/vorbis/>`__]
# -  ``"AMR_NB"``: Adaptive Multi-Rate
#    [`wikipedia <https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_audio_codec>`__]
# -  ``"AMR_WB"``: Adaptive Multi-Rate Wideband
#    [`wikipedia <https://en.wikipedia.org/wiki/Adaptive_Multi-Rate_Wideband>`__]
# -  ``"OPUS"``: Opus [`opus-codec.org <https://opus-codec.org/>`__]
# -  ``"GSM"``: GSM-FR
#    [`wikipedia <https://en.wikipedia.org/wiki/Full_Rate>`__]
# -  ``"HTK"``: Single channel 16-bit PCM
# -  ``"UNKNOWN"`` None of above
#

######################################################################
# **Note**
#
# -  ``bits_per_sample`` can be ``0`` for formats with compression and/or
#    variable bit rate (such as MP3).
# -  ``num_frames`` can be ``0`` for GSM-FR format.
#

metadata = torchaudio.info(SAMPLE_GSM)
print(metadata)


######################################################################
# Querying file-like object
# -------------------------
#
# :py:func:`torchaudio.info` works on file-like objects.
#

url = "https://download.pytorch.org/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav"
with requests.get(url, stream=True) as response:
    metadata = torchaudio.info(_hide_seek(response.raw))
print(metadata)

######################################################################
# .. note::
#
#    When passing a file-like object, ``info`` does not read
#    all of the underlying data; rather, it reads only a portion
#    of the data from the beginning.
#    Therefore, for a given audio format, it may not be able to retrieve the
#    correct metadata, including the format itself. In such case, you
#    can pass ``format`` argument to specify the format of the audio.

######################################################################
# Loading audio data
# ------------------
#
# To load audio data, you can use :py:func:`load_torchcodec`.
#
# This function accepts a path-like object or file-like object as input.
#
# The returned value is a tuple of waveform (``Tensor``) and sample rate
# (``int``).
#
# By default, the resulting tensor object has ``dtype=torch.float32`` and
# its value range is ``[-1.0, 1.0]``.
#
# For the list of supported format, please refer to `the torchaudio
# documentation <https://pytorch.org/audio>`__.
#

waveform, sample_rate = load_torchcodec(SAMPLE_WAV)


######################################################################
#
def plot_waveform(waveform, sample_rate):
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
    figure.suptitle("waveform")


######################################################################
#
plot_waveform(waveform, sample_rate)


######################################################################
#
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)


######################################################################
#
plot_specgram(waveform, sample_rate)


######################################################################
#
Audio(waveform.numpy()[0], rate=sample_rate)

######################################################################
# Loading from file-like object
# -----------------------------
#
# The I/O functions support file-like objects.
# This allows for fetching and decoding audio data from locations
# within and beyond the local file system.
# The following examples illustrate this.
#

######################################################################
#

# Load audio data as HTTP request
url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
with requests.get(url, stream=False) as response:
    waveform, sample_rate = load_torchcodec(response.content)
plot_specgram(waveform, sample_rate, title="HTTP datasource")

######################################################################
#

# Load audio from tar file
tar_path = download_asset("tutorial-assets/VOiCES_devkit.tar.gz")
tar_item = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
with tarfile.open(tar_path, mode="r") as tarfile_:
    fileobj = tarfile_.extractfile(tar_item)
    waveform, sample_rate = load_torchcodec(fileobj)
plot_specgram(waveform, sample_rate, title="TAR file")

######################################################################
#

# Load audio from S3
bucket = "pytorch-tutorial-assets"
key = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
response = client.get_object(Bucket=bucket, Key=key)
waveform, sample_rate = load_torchcodec(BytesIO(response['Body'].read()))
plot_specgram(waveform, sample_rate, title="From S3")
