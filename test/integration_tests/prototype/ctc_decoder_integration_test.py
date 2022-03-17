import pytest
from torchaudio.prototype.ctc_decoder import lexicon_decoder, download_pretrained_files


@pytest.mark.parametrize("model", ["librispeech", "librispeech-3-gram"])
def test_download_pretrain(model):
    # smoke test for downloading pretrained files
    _ = download_pretrained_files(model)


@pytest.mark.parametrize("model", ["librispeech", "librispeech-3-gram"])
def test_decoder_from_pretrained(model):
    # smoke test for constructing decoder from pretrained files
    pretrained_files = download_pretrained_files(model)

    _ = lexicon_decoder(
        lexicon=pretrained_files.lexicon,
        tokens=pretrained_files.tokens,
        lm=pretrained_files.lm,
    )
