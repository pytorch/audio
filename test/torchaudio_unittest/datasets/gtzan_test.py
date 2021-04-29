import os
from pathlib import Path

from torchaudio.datasets import gtzan

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


def get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    mocked_samples = []
    mocked_training = []
    mocked_validation = []
    mocked_testing = []
    sample_rate = 22050

    seed = 0
    for genre in gtzan.gtzan_genres:
        base_dir = os.path.join(root_dir, 'genres', genre)
        os.makedirs(base_dir, exist_ok=True)
        for i in range(100):
            filename = f'{genre}.{i:05d}'
            path = os.path.join(base_dir, f'{filename}.wav')
            data = get_whitenoise(sample_rate=sample_rate, duration=0.01, n_channels=1, dtype='int16', seed=seed)
            save_wav(path, data, sample_rate)
            sample = (normalize_wav(data), sample_rate, genre)
            mocked_samples.append(sample)
            if filename in gtzan.filtered_test:
                mocked_testing.append(sample)
            if filename in gtzan.filtered_train:
                mocked_training.append(sample)
            if filename in gtzan.filtered_valid:
                mocked_validation.append(sample)
            seed += 1
    return (mocked_samples, mocked_training, mocked_validation, mocked_testing)


class TestGTZAN(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = None
    samples = []
    training = []
    validation = []
    testing = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        mocked_data = get_mock_dataset(cls.root_dir)
        cls.samples = mocked_data[0]
        cls.training = mocked_data[1]
        cls.validation = mocked_data[2]
        cls.testing = mocked_data[3]

    def test_no_subset(self):
        dataset = gtzan.GTZAN(self.root_dir)

        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.samples[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[i][1]
            assert label == self.samples[i][2]
            n_ite += 1
        assert n_ite == len(self.samples)

    def _test_training(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.training[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.training[i][1]
            assert label == self.training[i][2]
            n_ite += 1
        assert n_ite == len(self.training)

    def _test_validation(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.validation[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.validation[i][1]
            assert label == self.validation[i][2]
            n_ite += 1
        assert n_ite == len(self.validation)

    def _test_testing(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.testing[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.testing[i][1]
            assert label == self.testing[i][2]
            n_ite += 1
        assert n_ite == len(self.testing)

    def test_training_str(self):
        train_dataset = gtzan.GTZAN(self.root_dir, subset='training')
        self._test_training(train_dataset)

    def test_validation_str(self):
        val_dataset = gtzan.GTZAN(self.root_dir, subset='validation')
        self._test_validation(val_dataset)

    def test_testing_str(self):
        test_dataset = gtzan.GTZAN(self.root_dir, subset='testing')
        self._test_testing(test_dataset)

    def test_training_path(self):
        root_dir = Path(self.root_dir)
        train_dataset = gtzan.GTZAN(root_dir, subset='training')
        self._test_training(train_dataset)

    def test_validation_path(self):
        root_dir = Path(self.root_dir)
        val_dataset = gtzan.GTZAN(root_dir, subset='validation')
        self._test_validation(val_dataset)

    def test_testing_path(self):
        root_dir = Path(self.root_dir)
        test_dataset = gtzan.GTZAN(root_dir, subset='testing')
        self._test_testing(test_dataset)
