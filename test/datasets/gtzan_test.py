import os

from torchaudio.datasets import gtzan

from ..common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)


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
        sample_rate = 22050
        seed = 0
        for genre in gtzan.gtzan_genres:
            base_dir = os.path.join(cls.root_dir, 'genres', genre)
            os.makedirs(base_dir, exist_ok=True)
            for i in range(100):
                filename = f'{genre}.{i:05d}'
                path = os.path.join(base_dir, f'{filename}.wav')
                data = get_whitenoise(sample_rate=sample_rate, duration=0.01, n_channels=1, dtype='int16', seed=seed)
                save_wav(path, data, sample_rate)
                sample = (normalize_wav(data), sample_rate, genre)
                cls.samples.append(sample)
                if filename in gtzan.filtered_test:
                    cls.testing.append(sample)
                if filename in gtzan.filtered_train:
                    cls.training.append(sample)
                if filename in gtzan.filtered_valid:
                    cls.validation.append(sample)
                seed += 1

    def test_no_subset(self):
        dataset = gtzan.GTZAN(self.root_dir)

        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.samples[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.samples[i][1]
            assert label == self.samples[i][2]
            n_ite += 1
        assert n_ite == len(self.samples)

    def test_training(self):
        dataset = gtzan.GTZAN(self.root_dir, subset='training')

        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.training[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.training[i][1]
            assert label == self.training[i][2]
            n_ite += 1
        assert n_ite == len(self.training)

    def test_validation(self):
        dataset = gtzan.GTZAN(self.root_dir, subset='validation')

        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.validation[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.validation[i][1]
            assert label == self.validation[i][2]
            n_ite += 1
        assert n_ite == len(self.validation)

    def test_testing(self):
        dataset = gtzan.GTZAN(self.root_dir, subset='testing')

        n_ite = 0
        for i, (waveform, sample_rate, label) in enumerate(dataset):
            self.assertEqual(waveform, self.testing[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == self.testing[i][1]
            assert label == self.testing[i][2]
            n_ite += 1
        assert n_ite == len(self.testing)
