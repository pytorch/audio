import os
from collections import defaultdict

from parameterized import parameterized
from torchaudio.prototype.datasets import Musan
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase


_SUBSET_TO_SUBDIRS = {
    "music": ["fma", "fma-western-art", "hd-classical", "jamendo", "rfm"],
    "noise": ["free-sound", "sound-bible"],
    "speech": ["librivox", "us-gov"],
}
_SAMPLE_RATE = 16_000


def _get_mock_dataset(dataset_dir):
    """
    Creates the following directory structure:
        music
            fma
            fma-western-art
            hd-classical
            jamendo
            rfm
        noise
            free-sound
            sound-bible
        speech
            librivox
            us-gov

    Then, within each leaf subdirectory, adds a WAV file containing white noise @ 16KHz.
    """
    mocked_samples = {}

    seed = 0
    os.makedirs(dataset_dir, exist_ok=True)
    for subset, subdirs in _SUBSET_TO_SUBDIRS.items():
        subset_samples = defaultdict(dict)
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_dir, subset, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            file_name = f"{subset}_{subdir}.wav"
            file_path = os.path.join(subdir_path, file_name)

            data = get_whitenoise(sample_rate=_SAMPLE_RATE, duration=10.00, n_channels=1, dtype="float32", seed=seed)
            save_wav(file_path, data, _SAMPLE_RATE)
            subset_samples[file_name] = (data, file_path)

            seed += 1
        mocked_samples[subset] = subset_samples
    return mocked_samples


class MusanTest(TempDirMixin, TorchaudioTestCase):
    @classmethod
    def setUpClass(cls):
        dataset_dir = os.path.join(cls.get_base_temp_dir(), "musan")
        cls.samples = _get_mock_dataset(dataset_dir)

    @parameterized.expand([("music",), ("noise",), ("speech",)])
    def test_musan(self, subset):
        dataset = Musan(self.get_base_temp_dir(), subset)
        for data, sample_rate, file_name in dataset:
            self.assertTrue(file_name in self.samples[subset])
            self.assertEqual(data, self.samples[subset][file_name][0])
            self.assertEqual(sample_rate, _SAMPLE_RATE)

    @parameterized.expand([("music",), ("noise",), ("speech",)])
    def test_musan_metadata(self, subset):
        dataset = Musan(self.get_base_temp_dir(), subset)
        for idx in range(len(dataset)):
            file_path, sample_rate, file_name = dataset.get_metadata(idx)
            self.assertTrue(file_name in self.samples[subset])
            self.assertEqual(file_path, self.samples[subset][file_name][1])
            self.assertEqual(sample_rate, _SAMPLE_RATE)
