import os
from pathlib import Path

from torchaudio.datasets import yesno

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


def get_mock_data(root_dir, labels):
    """
    root_dir: path
    labels: list of labels
    """
    mocked_data = []
    base_dir = os.path.join(root_dir, 'waves_yesno')
    os.makedirs(base_dir, exist_ok=True)
    for i, label in enumerate(labels):
        filename = f'{"_".join(str(l) for l in label)}.wav'
        path = os.path.join(base_dir, filename)
        data = get_whitenoise(sample_rate=8000, duration=6, n_channels=1, dtype='int16', seed=i)
        save_wav(path, data, 8000)
        mocked_data.append(normalize_wav(data))
    return mocked_data


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
        cls.data = get_mock_data(cls.root_dir, cls.labels)

    def _test_yesno(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            expected_label = self.labels[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 8000
            assert label == expected_label
            n_ite += 1
        assert n_ite == len(self.data)

    def test_yesno_str(self):
        dataset = yesno.YESNO(self.root_dir)
        self._test_yesno(dataset)

    def test_yesno_path(self):
        dataset = yesno.YESNO(Path(self.root_dir))
        self._test_yesno(dataset)
