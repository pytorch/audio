import pytest
import torch
import torchaudio
from torchaudio.prototype.pipelines import CONVTASNET_BASE_LIBRI2MIX


@pytest.mark.parametrize(
    "bundle,task",
    [
        [CONVTASNET_BASE_LIBRI2MIX, "speech_separation"],
    ],
)
def test_source_separation_models(bundle, task, mixture_speech, expected_tensor):
    """Smoke test of source separation pipeline"""
    separator = bundle.get_separator()
    mixture_speech, sample_rate = torchaudio.load(mixture_speech)
    expected_tensor = torch.load(expected_tensor)
    mixture_speech = mixture_speech.reshape(1, 1, -1)
    estimated_sources = separator(mixture_speech)
    expected_spectrogram = torchaudio.functional.spectrogram(
        expected_tensor,
        n_fft=400,
        hop_length=160,
        window=torch.ones(400),
        win_length=400,
        pad=False,
        normalized=False,
        power=2,
    )
    estimated_spectrogram = torchaudio.functional.spectrogram(
        estimated_sources,
        n_fft=400,
        hop_length=160,
        window=torch.ones(400),
        win_length=400,
        pad=False,
        normalized=False,
        power=2,
    )
    expected_peak_frequency = expected_spectrogram.argmax(dim=2)
    estimated_peak_frequency = estimated_spectrogram.argmax(dim=2)
    assert torch.equal(expected_peak_frequency, estimated_peak_frequency) is True
