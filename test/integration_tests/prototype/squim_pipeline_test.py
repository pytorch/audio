import pytest
import torchaudio
from torchaudio.prototype.pipelines import SQUIM_OBJECTIVE

@pytest.mark.parametrize(
    "lang,expected",
    [
        ("en", [0.9978380799293518, 4.23893404006958, 24.217193603515625]),
    ]
)
def test_squim_objective_pretrained_weights(lang, expected, sample_speech):
    """Test that the metric scores estimated by SquimObjective Bundle is identical to the expected result.
    """
    bundle = SQUIM_OBJECTIVE

    # Get SquimObjective model
    model = bundle.get_model()
    # Create a synthetic waveform
    waveform, sample_rate = torchaudio.load(sample_speech)
    scores = model(waveform)
    for i in range(3):
        assert scores[i].item() == expected[i]
