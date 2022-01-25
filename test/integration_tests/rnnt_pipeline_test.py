import pytest
import torchaudio
from torchaudio.prototype.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH


@pytest.mark.parametrize(
    "bundle,lang,expected",
    [
        (
            EMFORMER_RNNT_BASE_LIBRISPEECH,
            "en",
            # SentencePiece token ids for "i have that curiosity beside me at this moment"
            [4096, 18, 135, 60, 3526, 1931, 113, 100, 138, 619],
        )
    ],
)
def test_rnnt(bundle, sample_speech, expected):
    feature_extractor = bundle.get_feature_extractor()
    decoder = bundle.get_decoder().eval()
    waveform, _ = torchaudio.load(sample_speech)
    features, length = feature_extractor(waveform.squeeze())
    hypotheses = decoder(features, length, 10)
    assert hypotheses[0].tokens == expected
