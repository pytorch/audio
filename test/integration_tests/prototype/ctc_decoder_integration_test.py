import pytest


@pytest.mark.parametrize("model", ["librispeech", "librispeech-3-gram"])
def test_decoder_from_pretrained(model, emissions):
    # smoke test for constructing and running decoder from pretrained files

    from torchaudio.prototype.ctc_decoder import lexicon_decoder, download_pretrained_files

    pretrained_files = download_pretrained_files(model)
    decoder = lexicon_decoder(
        lexicon=pretrained_files.lexicon,
        tokens=pretrained_files.tokens,
        lm=pretrained_files.lm,
    )
    decoder(emissions)
