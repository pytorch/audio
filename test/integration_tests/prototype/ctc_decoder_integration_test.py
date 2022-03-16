from parameterized import parameterized
from torchaudio_unittest.common_utils import skipIfNoCtcDecoder
from torchaudio_unittest.common_utils.ctc_decoder_utils import is_ctc_decoder_available


if is_ctc_decoder_available():
    from torchaudio.prototype.ctc_decoder import lexicon_decoder, download_pretrained_files


@skipIfNoCtcDecoder
@parameterized.expand(["librispeech", "librispeech-3-gram"])
def test_download_pretrain(model):
    # smoke test for downloading pretrained files
    _ = download_pretrained_files(model)


@skipIfNoCtcDecoder
@parameterized.expand(["librispeech", "librispeech-3-gram"])
def test_decoder_from_pretrained(model):
    # smoke test for constructing decoder from pretrained files
    pretrained_files = download_pretrained_files(model)

    _ = lexicon_decoder(
        lexicon=pretrained_files.lexicon,
        tokens=pretrained_files.tokens,
        lm=pretrained_files.lm,
    )
