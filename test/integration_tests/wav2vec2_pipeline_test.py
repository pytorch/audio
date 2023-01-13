import sys

import pytest
import torchaudio
from torchaudio.pipelines import (
    HUBERT_ASR_LARGE,
    HUBERT_ASR_XLARGE,
    HUBERT_BASE,
    HUBERT_LARGE,
    HUBERT_XLARGE,
    VOXPOPULI_ASR_BASE_10K_DE,
    VOXPOPULI_ASR_BASE_10K_EN,
    VOXPOPULI_ASR_BASE_10K_ES,
    VOXPOPULI_ASR_BASE_10K_FR,
    VOXPOPULI_ASR_BASE_10K_IT,
    WAV2VEC2_ASR_BASE_100H,
    WAV2VEC2_ASR_BASE_10M,
    WAV2VEC2_ASR_BASE_960H,
    WAV2VEC2_ASR_LARGE_100H,
    WAV2VEC2_ASR_LARGE_10M,
    WAV2VEC2_ASR_LARGE_960H,
    WAV2VEC2_ASR_LARGE_LV60K_100H,
    WAV2VEC2_ASR_LARGE_LV60K_10M,
    WAV2VEC2_ASR_LARGE_LV60K_960H,
    WAV2VEC2_BASE,
    WAV2VEC2_LARGE,
    WAV2VEC2_LARGE_LV60K,
    WAV2VEC2_XLSR53,
    WAV2VEC2_XLSR_1B,
    WAV2VEC2_XLSR_2B,
    WAV2VEC2_XLSR_300M,
    WAVLM_BASE,
    WAVLM_BASE_PLUS,
    WAVLM_LARGE,
)

sys.path.append("..")
from torchaudio_unittest.common_utils.case_utils import skipIfNotInCI


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
        WAVLM_BASE,
        WAVLM_BASE_PLUS,
        WAVLM_LARGE,
    ],
)
def test_pretraining_models(bundle):
    """Smoke test of downloading weights for pretraining models"""
    bundle.get_model()


@skipIfNotInCI
@pytest.mark.parametrize(
    "bundle",
    [
        WAV2VEC2_XLSR_300M,
        WAV2VEC2_XLSR_1B,
        WAV2VEC2_XLSR_2B,
    ],
)
def test_xlsr_pretraining_models(bundle):
    """Smoke test of downloading weights for pretraining models"""
    bundle.get_model()


@pytest.mark.parametrize(
    "bundle,lang,expected",
    [
        (WAV2VEC2_ASR_BASE_10M, "en", "I|HAD|THAT|CURIYOSSITY|BESID|ME|AT|THIS|MOMENT|"),
        (WAV2VEC2_ASR_BASE_100H, "en", "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (WAV2VEC2_ASR_BASE_960H, "en", "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (WAV2VEC2_ASR_LARGE_10M, "en", "I|HAD|THAT|CURIOUSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (WAV2VEC2_ASR_LARGE_100H, "en", "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (WAV2VEC2_ASR_LARGE_960H, "en", "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (WAV2VEC2_ASR_LARGE_LV60K_10M, "en", "I|HAD|THAT|CURIOUSITY|BESID|ME|AT|THISS|MOMENT|"),
        (WAV2VEC2_ASR_LARGE_LV60K_100H, "en", "I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (WAV2VEC2_ASR_LARGE_LV60K_960H, "en", "I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (HUBERT_ASR_LARGE, "en", "I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (HUBERT_ASR_XLARGE, "en", "I|HAVE|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|"),
        (
            VOXPOPULI_ASR_BASE_10K_EN,
            "en2",
            "i|hope|that|we|will|see|a|ddrasstic|decrease|of|funding|for|the|failed|eu|project|and|that|more|money|will|come|back|to|the|taxpayers",  # noqa: E501
        ),
        (
            VOXPOPULI_ASR_BASE_10K_ES,
            "es",
            "la|primera|que|es|imprescindible|pensar|a|pequeña|a|escala|para|implicar|y|complementar|así|la|actuación|global",  # noqa: E501
        ),
        (VOXPOPULI_ASR_BASE_10K_DE, "de", "dabei|spielt|auch|eine|sorgfältige|berichterstattung|eine|wichtige|rolle"),
        (
            VOXPOPULI_ASR_BASE_10K_FR,
            "fr",
            "la|commission|va|faire|des|propositions|sur|ce|sujet|comment|mettre|en|place|cette|capacité|fiscale|et|le|conseil|européen|y|reviendra|sour|les|sujets|au|moins|de|mars",  # noqa: E501
        ),
        (
            VOXPOPULI_ASR_BASE_10K_IT,
            "it",
            "credo|che|illatino|non|sia|contemplato|tra|le|traduzioni|e|quindi|mi|attengo|allitaliano",
        ),
    ],
)
def test_finetune_asr_model(
    bundle,
    lang,
    expected,
    sample_speech,
    ctc_decoder,
):
    """Smoke test of downloading weights for fine-tuning models and simple transcription"""
    model = bundle.get_model().eval()
    waveform, sample_rate = torchaudio.load(sample_speech)
    emission, _ = model(waveform)
    decoder = ctc_decoder(bundle.get_labels())
    result = decoder(emission[0])
    assert result == expected
