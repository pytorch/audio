import torchaudio
from torchaudio.models import (
    hubert_base,
    hubert_large,
    hubert_xlarge,
    hubert_ft_large,
    hubert_ft_xlarge,
)
import pytest


@pytest.mark.parametrize(
    "factory_func,checkpoint",
    [
        (hubert_base, 'fairseq_ls960'),
        (hubert_large, 'fairseq_ll60k'),
        (hubert_xlarge, 'fairseq_ll60k')
    ]
)
def test_pretraining_models(factory_func, checkpoint):
    """Smoke test of downloading weights for pretraining models"""
    factory_func(checkpoint=checkpoint)


@pytest.mark.parametrize(
    "factory_func,checkpoint,expected",
    [
        (hubert_ft_large, 'fairseq_ll60k_asr_ls960', 'I HAVE THAT CURIOSITY BESIDE ME AT THIS MOMENT '),
        (hubert_ft_xlarge, 'fairseq_ll60k_asr_ls960', 'I HAVE THAT CURIOSITY BESIDE ME AT THIS MOMENT ')
    ]
)
def test_finetune_asr_model(
        factory_func,
        checkpoint,
        expected,
        sample_speech_16000_en,
        ctc_decoder_en,
):
    """Smoke test of downloading weights for fine-tuning models and simple transcription"""
    model = factory_func(checkpoint=checkpoint).eval()
    waveform, sample_rate = torchaudio.load(sample_speech_16000_en)
    emission, _ = model(waveform)
    result = ctc_decoder_en(emission[0])
    assert result == expected
