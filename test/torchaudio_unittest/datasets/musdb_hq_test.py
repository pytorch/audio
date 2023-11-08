import os

import torch
from parameterized import parameterized
from torchaudio.datasets import musdb_hq
from torchaudio.datasets.musdb_hq import _VALIDATION_SET
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase

_SOURCE_SETS = [
    (None,),
    (["bass", "drums", "other", "vocals"],),
    (["bass", "drums", "other"],),
    (["bass", "drums", "vocals"],),
    (["bass", "vocals", "other"],),
    (["vocals", "drums", "other"],),
    (["mixture"],),
]
seed_dict = {
    "bass": 0,
    "drums": 1,
    "other": 2,
    "mixture": 3,
    "vocals": 4,
}
EXT = ".wav"


def _save_sample(dataset_dir, folder, song, source, sample_rate, seed):
    # create and save audio samples to corresponding files
    path = os.path.join(dataset_dir, folder)
    os.makedirs(path, exist_ok=True)
    song_path = os.path.join(path, str(song))
    os.makedirs(song_path, exist_ok=True)
    source_path = os.path.join(song_path, f"{source}{EXT}")

    data = get_whitenoise(
        sample_rate=sample_rate,
        duration=5,
        n_channels=2,
        seed=seed,
    )
    save_wav(source_path, data, sample_rate)

    sample = (data, sample_rate, 5 * sample_rate, song)

    return sample


def _get_mocked_samples(dataset_dir, sample_rate):
    sample_count = 5

    all_samples = {"train": {}, "test": {}}

    folders = ["train", "test"]
    sources = ["bass", "drums", "other", "mixture", "vocals"]

    curr_idx = 0
    for folder in folders:
        for _ in range(sample_count):
            sample_list = []
            for source in sources:
                sample = _save_sample(dataset_dir, folder, str(curr_idx), source, sample_rate, seed_dict.get(source))
                sample_list.append(sample)
            all_samples[folder][str(curr_idx)] = sample_list
            curr_idx += 1
        if folder == "train":
            for name in _VALIDATION_SET:
                sample_list = []
                for source in sources:
                    sample = _save_sample(dataset_dir, folder, name, source, sample_rate, seed_dict.get(source))
                    sample_list.append(sample)
                all_samples[folder][name] = sample_list

    return all_samples


def get_mock_dataset(dataset_dir):
    """
    dataset_dir: directory to the mocked dataset
    """
    os.makedirs(dataset_dir, exist_ok=True)
    sample_rate = 44100

    return _get_mocked_samples(dataset_dir, sample_rate)


class TestMusDB_HQ(TempDirMixin, TorchaudioTestCase):
    root_dir = None

    train_all_samples = {}
    train_only_samples = {}
    validation_samples = {}
    test_samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(cls.root_dir, "musdb18hq")
        full_dataset = get_mock_dataset(dataset_dir)
        cls.train_all_samples = full_dataset["train"]
        cls.test_samples = full_dataset["test"]
        for key in cls.train_all_samples:
            if key in _VALIDATION_SET:
                cls.validation_samples[key] = cls.train_all_samples[key]
            else:
                cls.train_only_samples[key] = cls.train_all_samples[key]

    def _test_musdb_hq(self, dataset, data_samples, sources):
        num_samples = 0
        for _, (data, sample_rate, num_frames, name) in enumerate(dataset):
            self.assertEqual(data, self.extractSources(data_samples[name], sources))
            assert sample_rate == data_samples[name][0][1]
            assert num_frames == data_samples[name][0][2]
            assert name == data_samples[name][0][3]
            num_samples += 1

        assert num_samples == len(data_samples)

    @parameterized.expand(_SOURCE_SETS)
    def testMusDBSources_train_all(self, sources):
        dataset = musdb_hq.MUSDB_HQ(self.root_dir, sources=sources, subset="train")
        self._test_musdb_hq(dataset, self.train_all_samples, sources)

    @parameterized.expand(_SOURCE_SETS)
    def testMusDBSources_train_with_validation(self, sources):
        dataset = musdb_hq.MUSDB_HQ(
            self.root_dir,
            sources=sources,
            subset="train",
            split="train",
        )
        self._test_musdb_hq(dataset, self.train_only_samples, sources)

    @parameterized.expand(_SOURCE_SETS)
    def testMusDBSources_validation(self, sources):
        dataset = musdb_hq.MUSDB_HQ(
            self.root_dir,
            sources=sources,
            subset="train",
            split="validation",
        )
        self._test_musdb_hq(dataset, self.validation_samples, sources)

    @parameterized.expand(_SOURCE_SETS)
    def testMusDBSources_test(self, sources):
        dataset = musdb_hq.MUSDB_HQ(
            self.root_dir,
            sources=sources,
            subset="test",
        )
        self._test_musdb_hq(dataset, self.test_samples, sources)

    def extractSources(self, samples, sources):
        sources = ["bass", "drums", "other", "vocals"] if not sources else sources
        return torch.stack([samples[seed_dict[source]][0] for source in sources])
