import pytest
import torchaudio
from torchaudio.utils import load_torchcodec
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE


@pytest.mark.parametrize(
    "lang,expected",
    [
        ("en", [0.9978380799293518, 4.23893404006958, 24.217193603515625]),
    ],
)
def test_squim_objective_pretrained_weights(lang, expected, sample_speech):
    """Test that the metric scores estimated by SquimObjective Bundle is identical to the expected result."""
    bundle = SQUIM_OBJECTIVE

    # Get SquimObjective model
    model = bundle.get_model()
    # Create a synthetic waveform
    waveform, sample_rate = load_torchcodec(sample_speech)
    scores = model(waveform)
    for i in range(3):
        assert abs(scores[i].item() - expected[i]) < 1e-5


@pytest.mark.parametrize(
    "task,expected",
    [
        ("speech_separation", [3.9257140159606934, 3.9391300678253174]),
    ],
)
def test_squim_subjective_pretrained_weights(task, expected, mixture_source, clean_sources):
    """Test that the metric scores estimated by SquimSubjective Bundle is identical to the expected result."""
    bundle = SQUIM_SUBJECTIVE

    # Get SquimObjective model
    model = bundle.get_model()
    # Load input mixture audio
    waveform, sample_rate = load_torchcodec(mixture_source)
    for i, source in enumerate(clean_sources):
        # Load clean reference
        clean_waveform, sample_rate = load_torchcodec(source)
        score = model(waveform, clean_waveform)
        assert abs(score.item() - expected[i]) < 1e-5
