import math

import torch
import torchaudio
from torchaudio.prototype.functional import oscillator_bank
from torchaudio.prototype.pipelines import HIFIGAN_VOCODER_V3_LJSPEECH


def test_hifi_gan_pretrained_weights():
    """Test that a waveform reconstructed from mel spectrogram by HiFiGAN bundle is close enough to the original.
    The main transformations performed in this test can be represented as
        - audio -> reference log mel spectrogram
        - audio -> mel spectrogram -> audio -> estimated log mel spectrogram
    In the end, we compare estimated log mel spectrogram to the reference one. See comments in code for details.
    """
    bundle = HIFIGAN_VOCODER_V3_LJSPEECH

    # Get HiFiGAN-compatible transformation from waveform to mel spectrogram
    mel_transform = bundle.get_mel_transform()
    # Get HiFiGAN vocoder
    vocoder = bundle.get_vocoder()
    # Create a synthetic waveform
    ref_waveform = get_sin_sweep(sample_rate=bundle.sample_rate, length=100000)
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
    # Check that relative MSE is below 4%
    mse = ((estimated_spectorogram - ref_spectrogram) ** 2).mean()
    mean_ref = ((ref_spectrogram) ** 2).mean()
    print(mse / mean_ref)
    assert mse / mean_ref < 0.04


def get_sin_sweep(sample_rate, length):
    """Create a waveform which changes frequency from 0 to the Nyquist frequency (half of the sample rate)"""
    nyquist_freq = sample_rate / 2
    freq = torch.logspace(0, math.log(0.99 * nyquist_freq, 10), length).unsqueeze(-1)
    amp = torch.ones((length, 1))

    waveform = oscillator_bank(freq, amp, sample_rate=sample_rate)
    return waveform.unsqueeze(0)
