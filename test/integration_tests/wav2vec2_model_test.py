import torchaudio
from torchaudio.models import (
    HUBERT_BASE,
    HUBERT_LARGE,
    HUBERT_XLARGE,
    HUBERT_ASR_LARGE,
    HUBERT_ASR_XLARGE,
)
import pytest


@pytest.mark.parametrize(
    "bundle",
    [
        HUBERT_BASE,
        HUBERT_LARGE,
        HUBERT_XLARGE,
    ]
)
def test_pretraining_models(bundle):
    """Smoke test of downloading weights for pretraining models"""
    bundle.get_model()


@pytest.mark.parametrize(
    "bundle,expected",
    [
        (HUBERT_ASR_LARGE, 'I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|'),
        (HUBERT_ASR_XLARGE, 'I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|')
    ]
)
def test_finetune_asr_model(
        bundle,
        expected,
        sample_speech_16000_en,
        ctc_decoder,
):
    """Smoke test of downloading weights for fine-tuning models and simple transcription"""
    model = bundle.get_model().eval()
    waveform, sample_rate = torchaudio.load(sample_speech_16000_en)
    emission, _ = model(waveform)
    decoder = ctc_decoder(bundle.labels)
    result = decoder(emission[0])
    assert result == expected
