import os
import sys

import pytest
import torch
import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX, HDEMUCS_HIGH_MUSDB, HDEMUCS_HIGH_MUSDB_PLUS


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
from source_separation.utils.metrics import sdr


@pytest.mark.parametrize(
    "bundle,task,channel,expected_score",
    [
        [CONVTASNET_BASE_LIBRI2MIX, "speech_separation", 1, 8.1373],
        [HDEMUCS_HIGH_MUSDB_PLUS, "music_separation", 2, 8.7480],
        [HDEMUCS_HIGH_MUSDB, "music_separation", 2, 8.0697],
    ],
)
def test_source_separation_models(bundle, task, channel, expected_score, mixture_source, clean_sources):
    """Integration test for the source separation pipeline.
    Given the mixture waveform with dimensions `(batch, channel, time)`, the pre-trained pipeline generates
    the separated sources Tensor with dimensions `(batch, num_sources, time)`.
    The test computes the scale-invariant signal-to-distortion ratio (Si-SDR) score in decibel (dB).
    Si-SDR score should be equal to or larger than the expected score.
    """
    model = bundle.get_model()
    mixture_waveform, sample_rate = torchaudio.load(mixture_source)
    assert sample_rate == bundle.sample_rate, "The sample rate of audio must match that in the bundle."
    clean_waveforms = []
    for source in clean_sources:
        clean_waveform, sample_rate = torchaudio.load(source)
        assert sample_rate == bundle.sample_rate, "The sample rate of audio must match that in the bundle."
        clean_waveforms.append(clean_waveform)
    mixture_waveform = mixture_waveform.reshape(1, channel, -1)
    estimated_sources = model(mixture_waveform)
    clean_waveforms = torch.cat(clean_waveforms).unsqueeze(0)
    estimated_sources = estimated_sources.reshape(1, -1, clean_waveforms.shape[-1])
    sdr_values = sdr(estimated_sources, clean_waveforms).mean()
    assert sdr_values >= expected_score
