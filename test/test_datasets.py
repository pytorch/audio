import os
import unittest

from torchaudio.datasets.commonvoice import COMMONVOICE
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchaudio.datasets.utils import diskcache_iterator, bg_iterator
from torchaudio.datasets.vctk import VCTK
from torchaudio.datasets.yesno import YESNO
from torchaudio.datasets.ljspeech import LJSPEECH
from torchaudio.datasets.gtzan import GTZAN
from torchaudio.datasets.cmuarctic import CMUARCTIC
from torchaudio.datasets.libritts import LIBRITTS

from .common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_asset_path,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


class TestDatasets(TorchaudioTestCase):
    backend = 'default'
    path = get_asset_path()

    def test_vctk(self):
        data = VCTK(self.path)
        data[0]

    def test_librispeech(self):
        data = LIBRISPEECH(self.path, "dev-clean")
        data[0]

    def test_ljspeech(self):
        data = LJSPEECH(self.path)
        data[0]

    def test_speechcommands(self):
        data = SPEECHCOMMANDS(self.path)
        data[0]

    def test_gtzan(self):
        data = GTZAN(self.path)
        data[0]

    def test_cmuarctic(self):
        data = CMUARCTIC(self.path)
        data[0]

    def test_libritts(self):
        data = LIBRITTS(self.path)
        data[0]


class TestCommonVoice(TorchaudioTestCase):
    backend = 'default'
    path = get_asset_path()

    def test_commonvoice(self):
        data = COMMONVOICE(self.path, url="tatar")
        data[0]

    def test_commonvoice_diskcache(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = diskcache_iterator(data)
        # Save
        data[0]
        # Load
        data[0]

    def test_commonvoice_bg(self):
        data = COMMONVOICE(self.path, url="tatar")
        data = bg_iterator(data, 5)
        for _ in data:
            pass


class TestYesNo(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    data = []
    labels = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        base_dir = os.path.join(cls.root_dir, 'waves_yesno')
        os.makedirs(base_dir, exist_ok=True)
        for label in cls.labels:
            filename = f'{"_".join(str(l) for l in label)}.wav'
            path = os.path.join(base_dir, filename)
            data = get_whitenoise(sample_rate=8000, duration=6, n_channels=1, dtype='int16')
            save_wav(path, data, 8000)
            cls.data.append(normalize_wav(data))

    def test_yesno(self):
        dataset = YESNO(self.root_dir)
        samples = list(dataset)
        samples.sort(key=lambda s: s[2])
        for i, (waveform, sample_rate, label) in enumerate(samples):
            expected_label = self.labels[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 8000
            assert label == expected_label


if __name__ == "__main__":
    unittest.main()
