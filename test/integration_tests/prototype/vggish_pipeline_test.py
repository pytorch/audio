import unittest

import torchaudio
from torchaudio.prototype.pipelines import VGGISH


class VGGishPipelineTest(unittest.TestCase):
    def test_vggish(self):
        input_sr = VGGISH.sample_rate
        input_proc = VGGISH.get_input_processor()
        model = VGGISH.get_model()
        path = torchaudio.utils.download_asset("test-assets/Chopin_Ballade_-1_In_G_Minor,_Op._23_excerpt.mp3")
        waveform, sr = torchaudio.load(path, backend="ffmpeg")
        waveform = waveform.mean(axis=0)
        waveform = torchaudio.functional.resample(waveform, sr, input_sr)
        batch = input_proc(waveform)
        assert batch.shape == (62, 1, 96, 64)
        output = model(batch)
        assert output.shape == (62, 128)
