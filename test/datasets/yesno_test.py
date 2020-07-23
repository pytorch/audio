import os

from torchaudio.datasets import yesno

from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


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
        dataset = yesno.YESNO(self.root_dir)
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            expected_label = self.labels[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 8000
            assert label == expected_label
