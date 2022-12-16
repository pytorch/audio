import os
import sys

import mir_eval

import pytest

import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.prototype.pipelines import HIFIGAN_GENERATOR_V3_LJSPEECH


def test_smoke_hifi_gan_bundle():
    """Smoke test of downloading weights for pretraining model"""
    HIFIGAN_GENERATOR_V3_LJSPEECH.get_vocoder()


@pytest.mark.parametrize(
    "lang",
    [
        ("en"),
        ("en2"),
    ],
)
def test_hifi_gan_pretrained_weights(lang, sample_speech):
    """Test that HiFiGAN bundle can reconstruct waveform from mel spectrogram with sufficient SDR score"""
    bundle = HIFIGAN_GENERATOR_V3_LJSPEECH

    # Get HiFiGAN-compatible transformation from waveform to mel spectrogram
    mel_transform = bundle.get_mel_transform()
    # Get HiFiGAN vocoder
    vocoder = bundle.get_vocoder()

    ref_waveform, ref_sample_rate = torchaudio.load(sample_speech)
    ref_waveform = F.resample(ref_waveform, orig_freq=ref_sample_rate, new_freq=bundle.sample_rate)
    ref_waveform = ref_waveform[:, : -(ref_waveform.shape[1] % mel_transform.hop_size)]

    # Generate mel spectrogram from waveform
    mel_spectrogram = mel_transform(ref_waveform)

    with torch.no_grad():
        # Generate waveform from mel spectrogram
        estimated_waveform = vocoder(mel_spectrogram).squeeze(0)
        sdr_value, sir, sar, perm = mir_eval.separation.bss_eval_sources(
            ref_waveform.numpy(), estimated_waveform.numpy()
        )
        # TODO: estimated_waveform is intelligible, but noisy and possibly shifted with respect to original,
        # so sdr_value is very low
        assert sdr_value > 8
