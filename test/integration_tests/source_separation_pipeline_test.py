import os
import sys

import pytest
import torch
import torchaudio
from torchaudio.prototype.pipelines import CONVTASNET_BASE_LIBRI2MIX


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
from source_separation.utils.metrics import PIT, sdr


@pytest.mark.parametrize(
    "bundle,task,expected_score",
    [
        [CONVTASNET_BASE_LIBRI2MIX, "speech_separation", 8.1374],
    ],
)
def test_source_separation_models(bundle, task, expected_score, mixture_source, clean_sources):
    """Smoke test of source separation pipeline"""
    separator = bundle.get_separator()
    mixture_waveform, _ = torchaudio.load(mixture_source)
    clean_waveforms = []
    for source in clean_sources:
        clean_waveform, _ = torchaudio.load(source)
        clean_waveforms.append(clean_waveform)
    mixture_waveform = mixture_waveform.reshape(1, 1, -1)
    estimated_sources = separator(mixture_waveform)
    clean_waveforms = torch.cat(clean_waveforms).unsqueeze(0)
    _sdr_pit = PIT(utility_func=sdr)
    sdr_values = _sdr_pit(estimated_sources, clean_waveforms)
    assert torch.isclose(sdr_values, torch.tensor([expected_score]))
