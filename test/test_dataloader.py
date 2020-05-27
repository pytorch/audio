import unittest

import torchaudio
from torch.utils.data import Dataset, DataLoader

from . import common_utils
from .common_utils import AudioBackendScope, BACKENDS


class TORCHAUDIODS(Dataset):
    def __init__(self):
        sound_files = ["sinewave.wav", "steam-train-whistle-daniel_simon.mp3"]
        self.data = [common_utils.get_asset_path(fn) for fn in sound_files]
        self.si, self.ei = torchaudio.info(common_utils.get_asset_path("sinewave.wav"))
        self.si.precision = 16
        self.E = torchaudio.sox_effects.SoxEffectsChain()
        self.E.append_effect_to_chain("rate", [self.si.rate])  # resample to 16000hz
        self.E.append_effect_to_chain("channels", [self.si.channels])  # mono signal
        self.E.append_effect_to_chain("trim", [0, "16000s"])  # first 16000 samples of audio

    def __getitem__(self, index):
        fn = self.data[index]
        self.E.set_input_file(fn)
        x, sr = self.E.sox_build_flow_effects()
        return x

    def __len__(self):
        return len(self.data)


class Test_DataLoader(unittest.TestCase):
    @unittest.skipIf("sox" not in BACKENDS, "sox not available")
    @AudioBackendScope("sox")
    def test_1(self):
        expected_size = (2, 1, 16000)
        ds = TORCHAUDIODS()
        dl = DataLoader(ds, batch_size=2)
        for x in dl:
            self.assertTrue(x.size() == expected_size)


if __name__ == '__main__':
    unittest.main()
