import pytest
import torchaudio
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH


@pytest.mark.parametrize(
    "bundle,lang,expected",
    [
        (EMFORMER_RNNT_BASE_LIBRISPEECH, "en", "i have that curiosity beside me at this moment"),
    ],
)
def test_rnnt(bundle, sample_speech, expected):
    feature_extractor = bundle.get_feature_extractor()
    decoder = bundle.get_decoder().eval()
    token_processor = bundle.get_token_processor()
    waveform, _ = torchaudio.load(sample_speech)
    features, length = feature_extractor(waveform.squeeze())
    hypotheses = decoder(features, length, 10)
    text = token_processor(hypotheses[0][0])
    assert text == expected
