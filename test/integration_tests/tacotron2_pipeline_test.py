import pytest
from torchaudio.pipelines import (
    TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH,
    TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH,
    TACOTRON2_WAVERNN_CHAR_LJSPEECH,
    TACOTRON2_WAVERNN_PHONE_LJSPEECH,
)


@pytest.mark.parametrize(
    "bundle",
    [
        TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH,
        TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH,
        TACOTRON2_WAVERNN_CHAR_LJSPEECH,
        TACOTRON2_WAVERNN_PHONE_LJSPEECH,
    ],
)
def test_tts_models(bundle):
    """Smoke test of TTS pipeline"""
    text = "Hello world! Text to Speech!"

    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2()
    vocoder = bundle.get_vocoder()
    processed, lengths = processor(text)
    mel_spec, lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(mel_spec, lengths)
