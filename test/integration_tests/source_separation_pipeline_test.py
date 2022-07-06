import pytest
import torch
import torchaudio
from torchaudio.prototype.pipelines import CONVTASNET_BASE_LIBRI2MIX


@pytest.mark.parametrize(
    "bundle",
    [
        CONVTASNET_BASE_LIBRI2MIX,
    ],
)
def test_tts_models(bundle, mixture_speech, expected_tensor):
    """Smoke test of source separation pipeline"""
    separator = bundle.get_separator()
    mixture_speech, sample_rate = torchaudio.load(mixture_speech)
    expected_tensor = torch.load(expected_tensor)
    mixture_speech = mixture_speech.reshape(1, 1, -1)
    estimated_sources = separator(mixture_speech)
    assert torch.equal(estimated_sources, expected_tensor) is True
