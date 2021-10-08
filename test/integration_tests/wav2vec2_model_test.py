import torchaudio
from torchaudio.models import (
    WAV2VEC2_BASE,
    WAV2VEC2_LARGE,
    WAV2VEC2_LARGE_LV60K,
    WAV2VEC2_ASR_BASE_10M,
    WAV2VEC2_ASR_BASE_100H,
    WAV2VEC2_ASR_BASE_960H,
    WAV2VEC2_ASR_LARGE_10M,
    WAV2VEC2_ASR_LARGE_100H,
    WAV2VEC2_ASR_LARGE_960H,
    WAV2VEC2_ASR_LARGE_LV60K_10M,
    WAV2VEC2_ASR_LARGE_LV60K_100H,
    WAV2VEC2_ASR_LARGE_LV60K_960H,
    WAV2VEC2_XLSR53,
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
        WAV2VEC2_BASE,
        WAV2VEC2_LARGE,
        WAV2VEC2_LARGE_LV60K,
        WAV2VEC2_XLSR53,
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
        (WAV2VEC2_ASR_BASE_10M, 'I|HAD|THAT|CURIYOSSITY|BESID|ME|AT|THIS|MOMENT|'),
        (WAV2VEC2_ASR_BASE_100H, 'I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|'),
        (WAV2VEC2_ASR_BASE_960H, 'I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|'),
        (WAV2VEC2_ASR_LARGE_10M, 'I|HAD|THAT|CURIOUSITY|BESIDE|ME|AT|THIS|MOMENT|'),
        (WAV2VEC2_ASR_LARGE_100H, 'I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|'),
        (WAV2VEC2_ASR_LARGE_960H, 'I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|'),
        (WAV2VEC2_ASR_LARGE_LV60K_10M, 'I|HAD|THAT|CURIOUSSITY|BESID|ME|AT|THISS|MOMENT|'),
        (WAV2VEC2_ASR_LARGE_LV60K_100H, 'I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|'),
        (WAV2VEC2_ASR_LARGE_LV60K_960H, 'I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|'),
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
