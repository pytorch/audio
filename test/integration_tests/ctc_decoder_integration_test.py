import pytest


@pytest.mark.parametrize(
    "model,expected",
    [
        ("librispeech", ["the", "captain", "shook", "his", "head"]),
        ("librispeech-3-gram", ["the", "captain", "shook", "his", "head"]),
    ],
)
def test_decoder_from_pretrained(model, expected, emissions):
    from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

    pretrained_files = download_pretrained_files(model)
    decoder = ctc_decoder(
        lexicon=pretrained_files.lexicon,
        tokens=pretrained_files.tokens,
        lm=pretrained_files.lm,
    )
    result = decoder(emissions)
    assert result[0][0].words == expected
