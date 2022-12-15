import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples"))
import torch
import torchaudio
import torchaudio.functional as F
from source_separation.utils.metrics import sdr
from torchaudio.prototype.pipelines import HIFIGAN_GENERATOR_V3_LJSPEECH

from torchaudio.transforms import MelSpectrogram


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
    # Config params from
    # https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/config_v3.json#L17-L27
    hop_size = 256
    n_fft = 1024
    win_length = 1024
    f_min, f_max = 0, 8000
    ref_waveform_raw, ref_sample_rate = torchaudio.load(sample_speech)
    ref_waveform = F.resample(ref_waveform_raw, orig_freq=ref_sample_rate, new_freq=bundle.sample_rate)
    # Generate mel spectrogram in the same way as the original HiFiGAN implementation does
    # https://github.com/jik876/hifi-gan/blob/4769534d45265d52a904b850da5a622601885777/meldataset.py#L49-L72
    mel_transform = MelSpectrogram(
        sample_rate=bundle.sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_size,
        f_min=f_min,
        f_max=f_max,
        n_mels=bundle._params["in_channels"],
        normalized=False,
        mel_scale="slaney",
        norm="slaney",
        center=True,
    )
    mel_spectrogram = mel_transform(ref_waveform)
    mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
    vocoder = bundle.get_vocoder()
    with torch.no_grad():
        estimated_waveform = vocoder(mel_spectrogram)
        ref_waveform = ref_waveform.unsqueeze(0)
        estimated_waveform = estimated_waveform[:, :, hop_size // 2 : -(hop_size // 2)]
        ref_waveform = ref_waveform[:, :, : -(ref_waveform.shape[2] % hop_size)]
        sdr_value = sdr(estimated_waveform, ref_waveform)
        # TODO: estimated_waveform is intelligible, but noisy and possibly shifted with respect to original,
        # so sdr_value is very low
        assert sdr_value > 8
