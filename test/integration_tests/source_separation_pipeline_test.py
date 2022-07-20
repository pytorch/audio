import os
import sys

import torch
import torchaudio
from torchaudio.prototype.pipelines import CONVTASNET_BASE_LIBRI2MIX


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
from source_separation.utils.metrics import PIT, sdr


def test_source_separation_models(mixture_source, clean_sources):
    """Integration test for the source separation pipeline.
    Given the mixture waveform with dimensions `(batch, 1, time)`, the pre-trained pipeline generates
    the separated sources Tensor with dimensions `(batch, num_sources, time)`.
    The test computes the scale-invariant signal-to-distortion ratio (Si-SDR) score in decibel (dB) with
    permutation invariant training (PIT) criterion. PIT computes Si-SDR scores between the estimated sources and the
    target sources for all permuations, then returns the highest values as the final output. The final
    Si-SDR score should be equal to or larger than the expected score.
    """
    BUNDLE = CONVTASNET_BASE_LIBRI2MIX
    EXPECTED_SCORE = 8.1373  # expected Si-SDR score.
    model = BUNDLE.get_model()
    mixture_waveform, sample_rate = torchaudio.load(mixture_source)
    assert sample_rate == BUNDLE.sample_rate, "The sample rate of audio must match that in the bundle."
    clean_waveforms = []
    for source in clean_sources:
        clean_waveform, sample_rate = torchaudio.load(source)
        assert sample_rate == BUNDLE.sample_rate, "The sample rate of audio must match that in the bundle."
        clean_waveforms.append(clean_waveform)
    mixture_waveform = mixture_waveform.reshape(1, 1, -1)
    estimated_sources = model(mixture_waveform)
    clean_waveforms = torch.cat(clean_waveforms).unsqueeze(0)
    _sdr_pit = PIT(utility_func=sdr)
    sdr_values = _sdr_pit(estimated_sources, clean_waveforms)
    assert sdr_values >= EXPECTED_SCORE
