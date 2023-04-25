"""
Torchaudio-Squim: Non-intrusive Speech Assessment in TorchAudio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""


######################################################################
# Author: `Anurag Kumar <anuragkr90@meta.com>`__, `Zhaoheng
# Ni <zni@meta.com>`__
# 


######################################################################
# 1. Overview
# ^^^^^^^^^^^
# 


######################################################################
# This tutorial shows uses of Torchaudio-Squim to estimate objective and
# subjective metrics for assessment of speech quality and intelligibility.
# 
# TorchAudio-Squim enables speech assessment in Torchaudio. It provides
# interface and pre-trained models to estimate various speech quality and
# intelligibility metrics. Currently, Torchaudio-Squim [1] supports
# reference-free estimation 3 widely used objective metrics:
# 
# -  Wideband Perceptual Estimation of Speech Quality (PESQ) [2]
# 
# -  Short-Time Objective Intelligibility (STOI) [3]
# 
# -  Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) [4]
# 
# It also supports estimation of subjective Mean Opinion Score (MOS) for a
# given audio waveform using Non-Matching References [1, 5].
# 
# **References**
# 
# [1] Kumar, Anurag, et al. “TorchAudio-Squim: Reference-less Speech
# Quality and Intelligibility measures in TorchAudio.” ICASSP 2023-2023
# IEEE International Conference on Acoustics, Speech and Signal Processing
# (ICASSP). IEEE, 2023.
# 
# [2] I. Rec, “P.862.2: Wideband extension to recommendation P.862 for the
# assessment of wideband telephone networks and speech codecs,”
# International Telecommunication Union, CH–Geneva, 2005.
# 
# [3] Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2010,
# March). A short-time objective intelligibility measure for
# time-frequency weighted noisy speech. In 2010 IEEE international
# conference on acoustics, speech and signal processing (pp. 4214-4217).
# IEEE.
# 
# [4] Le Roux, Jonathan, et al. “SDR–half-baked or well done?.” ICASSP
# 2019-2019 IEEE International Conference on Acoustics, Speech and Signal
# Processing (ICASSP). IEEE, 2019.
# 
# [5] Manocha, Pranay, and Anurag Kumar. “Speech quality assessment
# through MOS using non-matching references.” Interspeech, 2022.
# 

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


######################################################################
# 2. Preparation
# ^^^^^^^^^^^^^^
# 
# First import the modules and define the helper functions.
# 
# We will need torch, torchaudio to use Torchaudio-squim, Matplotlib to
# plot data, pystoi, pesq for computing reference metrics.
# 

try:
    from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE
    from torchaudio.prototype.pipelines import SQUIM_SUBJECTIVE
    from pesq import pesq
    from pystoi import stoi
except ImportError:
    import google.colab

    print(
        """
        To enable running this notebook in Google Colab, install nightly
        torch and torchaudio builds by adding the following code block to the top
        of the notebook before running it:
        !pip3 uninstall -y torch torchvision torchaudio
        !pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
        !pip3 install pesq
        !pip3 install pystoi
        """
    )


######################################################################
# 
# 

import torchaudio.functional as F
from torchaudio.utils import download_asset
from IPython.display import Audio
import matplotlib.pyplot as plt


def si_snr(estimate, reference, epsilon=1e-8):
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_pow = reference.pow(2).mean(axis=1, keepdim=True)
    mix_pow = (estimate * reference).mean(axis=1, keepdim=True)
    scale = mix_pow / (reference_pow + epsilon)

    reference = scale * reference
    error = estimate - reference

    reference_pow = reference.pow(2)
    error_pow = error.pow(2)

    reference_pow = reference_pow.mean(axis=1)
    error_pow = error_pow.mean(axis=1)

    si_snr = 10 * torch.log10(reference_pow) - 10 * torch.log10(error_pow)
    return si_snr.item()


def plot_waveform(waveform, title):
    wav_numpy = waveform.numpy()

    sample_size = waveform.shape[1]
    time_axis = torch.arange(0, sample_size) / 16000

    figure = plt.figure(figsize=(10,4))
    axes = figure.gca()
    axes.plot(time_axis, wav_numpy[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, title):
    wav_numpy = waveform.numpy()

    sample_size = waveform.shape[1]

    figure = plt.figure(figsize=(10,4))
    axes = figure.gca()
    axes.specgram(wav_numpy[0], Fs=sample_rate)
    figure.suptitle(title)
    plt.show(block=False)


######################################################################
# 3. Load Speech and Noise Sample
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

SAMPLE_SPEECH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo.wav")


######################################################################
# 
# 

WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH  = torchaudio.load(SAMPLE_SPEECH)
WAVEFORM_NOISE, SAMPLE_RATE_NOISE = torchaudio.load(SAMPLE_NOISE)
WAVEFORM_NOISE = WAVEFORM_NOISE[0:1,:]


######################################################################
# Currently, Torchaudio-Squim model only supports 16000 Hz sampling rate.
# Resample the waveforms necessary.
# 

if SAMPLE_RATE_SPEECH != 16000:
    WAVEFORM_SPEECH = F.resample(WAVEFORM_SPEECH, SAMPLE_RATE_SPEECH, 16000)

if SAMPLE_RATE_NOISE != 16000:
    WAVEFORM_NOISE = F.resample(WAVEFORM_NOISE, SAMPLE_RATE_NOISE, 16000)


######################################################################
# Trim waveforms so that they have the same number of frames.
# 

if WAVEFORM_SPEECH.shape[1] < WAVEFORM_NOISE.shape[1]:
    WAVEFORM_NOISE = WAVEFORM_NOISE[:, : WAVEFORM_SPEECH.shape[1]]
else:
    WAVEFORM_SPEECH = WAVEFORM_SPEECH[:, : WAVEFORM_NOISE.shape[1]]


######################################################################
# Play speech sample
# 

Audio(WAVEFORM_SPEECH.numpy()[0], rate=16000)


######################################################################
# Play noise sample
# 

Audio(WAVEFORM_NOISE.numpy()[0], rate=16000)


######################################################################
# 4. Create distorted (noisy) speech samples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

snr_dbs = torch.tensor([20, -5])
WAVEFORM_DISTORTED = F.add_noise(WAVEFORM_SPEECH, WAVEFORM_NOISE, snr_dbs)


######################################################################
# Play distorted speech with 20dB SNR
# 

Audio(WAVEFORM_DISTORTED.numpy()[0], rate=16000)


######################################################################
# Play distorted speech with -5dB SNR
# 

Audio(WAVEFORM_DISTORTED.numpy()[1], rate=16000)


######################################################################
# 4. Visualize the waveforms
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 


######################################################################
# Visualize speech sample
# 

plot_waveform(WAVEFORM_SPEECH, "Clean Speech")
plot_specgram(WAVEFORM_SPEECH, 16000,"Clean Speech Spectrogram")


######################################################################
# Visualize noise sample
# 

plot_waveform(WAVEFORM_NOISE, "Noise")
plot_specgram(WAVEFORM_NOISE, 16000, "Noise Spectrogram")


######################################################################
# Visualize distorted speech with 20dB SNR
# 

plot_waveform(WAVEFORM_DISTORTED[0:1], f"Distorted Speech with {snr_dbs[0]}dB SNR")
plot_specgram(WAVEFORM_DISTORTED[0:1], 16000, f"Distorted Speech with {snr_dbs[0]}dB SNR")


######################################################################
# Visualize distorted speech with -5dB SNR
# 

plot_waveform(WAVEFORM_DISTORTED[1:2], f"Distorted Speech with {snr_dbs[1]}dB SNR")
plot_specgram(WAVEFORM_DISTORTED[1:2], 16000, f"Distorted Speech with {snr_dbs[1]}dB SNR")


######################################################################
# 5. Predict Objective Metrics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 


######################################################################
# Compare model outputs with ground truths for distorted speech with 20dB
# SNR
# 

objective_model = SQUIM_OBJECTIVE.get_model()
stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED[0:1,:])
print(f"Estimated metrics for distorted speech at {snr_dbs[0]}dB are\n")
print(f"STOI: {stoi_hyp[0]}")
print(f"PESQ: {pesq_hyp[0]}")
print(f"SI-SDR: {si_sdr_hyp[0]}\n")

pesq_ref = pesq(16000, WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), mode="wb")
stoi_ref = stoi(WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[0].numpy(), 16000, extended=False)
si_sdr_ref = si_snr(WAVEFORM_DISTORTED[0:1], WAVEFORM_SPEECH)
print(f"Reference metrics for distorted speech at {snr_dbs[0]}dB are\n")
print(f"STOI: {stoi_ref}")
print(f"PESQ: {pesq_ref}")
print(f"SI-SDR: {si_sdr_ref}")


######################################################################
# Compare model outputs with ground truths for distorted speech with -5dB
# SNR
# 

objective_model = SQUIM_OBJECTIVE.get_model()
stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(WAVEFORM_DISTORTED[1:2,:])
print(f"Estimated metrics for distorted speech at {snr_dbs[1]}dB are\n")
print(f"STOI: {stoi_hyp[0]}")
print(f"PESQ: {pesq_hyp[0]}")
print(f"SI-SDR: {si_sdr_hyp[0]}\n")

pesq_ref = pesq(16000, WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[1].numpy(), mode="wb")
stoi_ref = stoi(WAVEFORM_SPEECH[0].numpy(), WAVEFORM_DISTORTED[1].numpy(), 16000, extended=False)
si_sdr_ref = si_snr(WAVEFORM_DISTORTED[1:2], WAVEFORM_SPEECH)
print(f"Reference metrics for distorted speech at {snr_dbs[1]}dB are\n")
print(f"STOI: {stoi_ref}")
print(f"PESQ: {pesq_ref}")
print(f"SI-SDR: {si_sdr_ref}")


######################################################################
# 5. Predict Mean Opinion Scores (Subjective)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 


######################################################################
# Load a non-matching reference (NMR)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 

NMR_SPEECH = download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")

WAVEFORM_NMR, SAMPLE_RATE_NMR = torchaudio.load(NMR_SPEECH)
if SAMPLE_RATE_NMR != 16000:
    WAVEFORM_NMR = F.resample(WAVEFORM_NMR, SAMPLE_RATE_NMR, 16000)

subjective_model = SQUIM_SUBJECTIVE.get_model()


######################################################################
# Compute MOS metric for distorted speech with 20dB SNR
# 

mos = subjective_model(WAVEFORM_DISTORTED[0:1,:], WAVEFORM_NMR)
print(f"Estimated MOS for distorted speech at {snr_dbs[0]}dB is MOS: {mos[0]}")


######################################################################
# Compute MOS metric for distorted speech with -5dB SNR
# 

mos = subjective_model(WAVEFORM_DISTORTED[1:2,:], WAVEFORM_NMR)
print(f"Estimated MOS for distorted speech at {snr_dbs[1]}dB is MOS: {mos[0]}")