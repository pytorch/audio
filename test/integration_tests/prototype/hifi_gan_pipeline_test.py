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
    """Test that a waveform reconstructed from mel spectrogram by HiFiGAN bundle is close enough to the original"""
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
    # Measure the reconstruction error.
    # Even though the reconstructed audio is perceptually very close to the original, it doesn't score well on
    # metrics like Si-SNR. It might be that HiFiGAN introduces non-uniform shifts to the reconstructed waveforms.
    # So to evaluate the recontruction error we compute mel spectrograms of the reference and recontructed waveforms,
    # and compare relative mean squared error of their logarithms.
    final_spec = torchaudio.transforms.MelSpectrogram(sample_rate=bundle.sample_rate, normalized=True)
    # Log mel spectrogram of the estimated waveform
    estimated_spectorogram = final_spec(estimated_waveform)
    estimated_spectorogram = torch.log(torch.clamp(estimated_spectorogram, min=1e-5))
    # Log mel spectrogram of the reference waveform
    ref_spectrogram = final_spec(ref_waveform)
    ref_spectrogram = torch.log(torch.clamp(ref_spectrogram, min=1e-5))
    # Check that relative MSE is below 2%
    mse = ((estimated_spectorogram - ref_spectrogram) ** 2).mean()
    mean_ref = ((ref_spectrogram) ** 2).mean()
    assert mse / mean_ref < 0.02
