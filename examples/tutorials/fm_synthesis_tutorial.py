"""
FM Synthesis
============

**Author**: `Moto Hira <moto@meta.com>`__

.. warning::
   This tutorial requires prototype DSP features, which are
   available in nightly builds.

   Please refer to https://pytorch.org/get-started/locally
   for instructions for installing a nightly build.

"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
#
from torchaudio.prototype.functional import oscillator_bank

import matplotlib.pyplot as plt
from IPython.display import Audio

######################################################################
#

SAMPLE_RATE = 44100
duration = 1.2
NUM_FRAMES = int(SAMPLE_RATE * duration)

F0 = 220.
F1 = 440.


def fm_synth(beta, f1=F1, f0=F0):
    print(f"Modulator Frequency: {f1} [Hz]")
    print(f"Carrior Frequency: {f0} [Hz]")
    print(f"Beta: {beta}")

    amp = torch.ones((NUM_FRAMES, 1), dtype=torch.float64)

    freq = torch.full((NUM_FRAMES, 1), f1, dtype=torch.float64)
    mod = oscillator_bank(freq, amp, sample_rate=SAMPLE_RATE, reduction="none")

    freq = f0 + beta * f0 * mod
    car = oscillator_bank(freq, amp, sample_rate=SAMPLE_RATE)
    return mod.sum(-1), car


######################################################################
#
def plot(mod, car, sample_rate=SAMPLE_RATE):
    fig, axes = plt.subplots(4, 1)
    axes[0].specgram(mod, Fs=sample_rate)
    spectrum, freqs, _, _ = axes[1].specgram(car, Fs=sample_rate)

    num_samples = int(SAMPLE_RATE * 0.01)
    t = torch.linspace(0, num_samples / sample_rate, num_samples)
    axes[2].plot(t, car[..., :num_samples])
    axes[3].plot(freqs, spectrum[:, spectrum.shape[1]//2], marker='+')


######################################################################
#
mod, car = fm_synth(beta=0.1)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=1)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=10)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=100)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=1, f1=0.01*F0)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=1, f1=0.1*F0)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=5, f1=0.1*F0)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=1, f1=1*F0)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=1, f1=10*F0)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
######################################################################
#
mod, car = fm_synth(beta=1, f1=100*F0)
plot(mod, car)
Audio(car, rate=SAMPLE_RATE)
