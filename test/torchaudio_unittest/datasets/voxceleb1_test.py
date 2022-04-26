import os

from torchaudio.datasets import voxceleb1
from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
)


def _save_sample(dataset_dir, sample_rate, speaker_id, youtube_id, idx, seed):
    # create and save audio samples to corresponding files
    # add random string before youtube_id
    youtube_id = "Zxhsj" + str(youtube_id)
    path = os.path.join(dataset_dir, "id10" + str(speaker_id), youtube_id)
    os.makedirs(path, exist_ok=True)
    filename = str(idx) + ".wav"
    file_path = os.path.join(path, filename)

    waveform = get_whitenoise(
        sample_rate=sample_rate,
        duration=0.01,
        n_channels=1,
        seed=seed,
    )
    save_wav(file_path, waveform, sample_rate)

    sample = (waveform, sample_rate, speaker_id, youtube_id)

    return sample


def _get_mocked_samples(dataset_dir, subset, sample_rate, seed):
    samples = []
    num_speakers = 3
    num_youtube = 5

    if subset == "dev":
        dataset_dir = os.path.join(dataset_dir, "vox1_dev_wav", "wav")
    elif subset == "test":
        dataset_dir = os.path.join(dataset_dir, "vox1_test_wav", "wav")
    else:
        raise ValueError(f"Expected 'dev' or 'test' for ``subset``. Found {subset}")

    idx = 0
    for speaker_id in range(num_speakers):
        for youtube_id in range(num_youtube):
            sample = _save_sample(dataset_dir, sample_rate, speaker_id, youtube_id, idx, seed)
            samples.append(sample)
            idx += 1
    return samples


def get_mock_dataset(dataset_dir):
    """
    dataset_dir: directory to the mocked dataset
    """
    os.makedirs(dataset_dir, exist_ok=True)
    sample_rate = 16000

    dev_seed = 0
    test_seed = 1

    mocked_dev_samples = _get_mocked_samples(dataset_dir, "dev", sample_rate, dev_seed)
    mocked_test_samples = _get_mocked_samples(dataset_dir, "test", sample_rate, test_seed)

    return (
        mocked_dev_samples,
        mocked_test_samples,
    )


class TestVoxCeleb1(TempDirMixin, TorchaudioTestCase):
    root_dir = None
    backend = "default"

    dev_samples = {}
    test_samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        (cls.dev_samples, cls.test_samples) = get_mock_dataset(cls.root_dir)

    def _testVoxCeleb1(self, dataset, data_samples):
        num_samples = 0
        for i, (waveform, sample_rate, speaker_id, youtube_id) in enumerate(dataset):
            self.assertEqual(waveform, data_samples[i][0])
            assert sample_rate == data_samples[i][1]
            assert speaker_id == data_samples[i][2]
            assert youtube_id == data_samples[i][3]
            num_samples += 1

        assert num_samples == len(data_samples)

    def testVoxCeleb1SubsetDev(self):
        dataset = voxceleb1.VoxCeleb1(self.root_dir, subset="dev")
        self._testVoxCeleb1(dataset, self.dev_samples)

    def testVoxCeleb1SubsetTest(self):
        dataset = voxceleb1.VoxCeleb1(self.root_dir, subset="test")
        self._testVoxCeleb1(dataset, self.test_samples)
