import os

from torchaudio.datasets import voxceleb1
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase


_NUM_SPEAKERS = 3
_NUM_YOUTUBE = 5


def _save_sample(dataset_dir: str, sample_rate: int, speaker_id: int, youtube_id: int, utterance_id: int, seed: int):
    """Create and save audio samples to corresponding files

    Args:
        dataset_dir (str): The directory of the dataset.
        sample_rate (int): Sample rate of waveform.
        speaker_id (int): The index of speaker sub directory.
        youtube_id (int): The index of youtube sub directory.
        utterance_id (int): The utterance index.
        seed (int): The seed to generate the waveform.

    Returns:
        Tuple[torch.Tensor, int, int, str, str]
        The waveform Tensor, sample rate, speaker label, file_name, and the file path.
    """
    # add random string before youtube_id
    youtube_id = "Zxhsj" + str(youtube_id)
    path = os.path.join(dataset_dir, "id10" + str(speaker_id), youtube_id)
    os.makedirs(path, exist_ok=True)
    filename = str(utterance_id) + ".wav"
    file_path = os.path.join(path, filename)
    waveform = get_whitenoise(
        sample_rate=sample_rate,
        duration=0.01,
        n_channels=1,
        seed=seed,
    )
    save_wav(file_path, waveform, sample_rate)
    file_name = "-".join(["id10" + str(speaker_id), youtube_id, str(utterance_id)])
    file_path = "/".join(["id10" + str(speaker_id), youtube_id, str(utterance_id) + ".wav"])
    return waveform, sample_rate, speaker_id, file_name, file_path


def get_mock_iden_dataset(root_dir: str, meta_file: str):
    """Get the mocked dataset for VoxCeleb1Identification dataset.

    Args:
        root_dir (str): Directory to the mocked dataset
        meta_file (str): The file name which stores the file list.

    Returns:
        Tuple[List, List, List]:
        The mocked samples for train, dev, and test subsets.
    """
    os.makedirs(root_dir, exist_ok=True)
    wav_dir = os.path.join(root_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    mocked_train_samples, mocked_dev_samples, mocked_test_samples = [], [], []
    sample_rate = 16000
    seed = 0
    idx = 1

    with open(os.path.join(root_dir, meta_file), "w") as f:
        for speaker_id in range(_NUM_SPEAKERS):
            for youtube_id in range(_NUM_YOUTUBE):
                waveform, sample_rate, speaker_id, file_name, file_path = _save_sample(
                    wav_dir, sample_rate, speaker_id, youtube_id, idx, seed
                )
                sample = (waveform, sample_rate, speaker_id, file_name)
                if idx % 1 == 0:
                    mocked_train_samples.append(sample)
                    f.write(f"1 {file_path}\n")
                elif idx % 2 == 0:
                    mocked_dev_samples.append(sample)
                    f.write(f"2 {file_path}\n")
                else:
                    mocked_test_samples.append(sample)
                    f.write(f"3 {file_path}\n")
                idx += 1
    return (
        mocked_train_samples,
        mocked_dev_samples,
        mocked_test_samples,
    )


def get_mock_veri_dataset(root_dir: str, meta_file: str):
    """Get the mocked dataset for VoxCeleb1Verification dataset.

    Args:
        root_dir (str): Directory to the mocked dataset
        meta_file (str): The file name which stores the file list.

    Returns:
        List[Sample]:
        The mocked samples.
    """
    os.makedirs(root_dir, exist_ok=True)
    wav_dir = os.path.join(root_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    mocked_samples = []
    sample_rate = 16000
    seed = 0
    idx = 1

    with open(os.path.join(root_dir, meta_file), "w") as f:
        for speaker_id1 in range(_NUM_SPEAKERS):
            for speaker_id2 in range(_NUM_SPEAKERS):
                for youtube_id in range(_NUM_YOUTUBE):
                    waveform_spk1, sample_rate, _, file_name_spk1, file_path_spk1 = _save_sample(
                        wav_dir, sample_rate, speaker_id1, youtube_id, idx, seed
                    )
                    waveform_spk2, sample_rate, _, file_name_spk2, file_path_spk2 = _save_sample(
                        wav_dir, sample_rate, speaker_id1, youtube_id, idx + 1, seed
                    )
                    if speaker_id1 == speaker_id2:
                        label = 1
                    else:
                        label = 0
                    sample = (waveform_spk1, waveform_spk2, sample_rate, label, file_name_spk1, file_name_spk2)
                    mocked_samples.append(sample)
                    f.write(f"{label} {file_path_spk1} {file_path_spk2}\n")
                    idx += 2
    return mocked_samples


class TestVoxCeleb1Identification(TempDirMixin, TorchaudioTestCase):
    root_dir = None

    meta_file = "iden_list.txt"
    train_samples = {}
    dev_samples = {}
    test_samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        (cls.train_samples, cls.dev_samples, cls.test_samples) = get_mock_iden_dataset(cls.root_dir, cls.meta_file)

    def _testVoxCeleb1Identification(self, dataset, data_samples):
        num_samples = 0
        for i, (waveform, sample_rate, speaker_id, file_id) in enumerate(dataset):
            self.assertEqual(waveform, data_samples[i][0])
            assert sample_rate == data_samples[i][1]
            assert speaker_id == data_samples[i][2]
            assert file_id == data_samples[i][3]
            num_samples += 1

        assert num_samples == len(data_samples)

    def testVoxCeleb1SubsetTrain(self):
        dataset = voxceleb1.VoxCeleb1Identification(self.root_dir, subset="train", meta_url=self.meta_file)
        self._testVoxCeleb1Identification(dataset, self.train_samples)

    def testVoxCeleb1SubsetDev(self):
        dataset = voxceleb1.VoxCeleb1Identification(self.root_dir, subset="dev", meta_url=self.meta_file)
        self._testVoxCeleb1Identification(dataset, self.dev_samples)

    def testVoxCeleb1SubsetTest(self):
        dataset = voxceleb1.VoxCeleb1Identification(self.root_dir, subset="test", meta_url=self.meta_file)
        self._testVoxCeleb1Identification(dataset, self.test_samples)


class TestVoxCeleb1Verification(TempDirMixin, TorchaudioTestCase):
    root_dir = None

    meta_file = "veri_test.txt"
    train_samples = {}
    dev_samples = {}
    test_samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        (cls.samples) = get_mock_veri_dataset(cls.root_dir, cls.meta_file)

    def testVoxCeleb1Verification(self):
        dataset = voxceleb1.VoxCeleb1Verification(self.root_dir, meta_url=self.meta_file)
        num_samples = 0
        for i, (waveform_spk1, waveform_spk2, sample_rate, label, file_id_spk1, file_id_spk2) in enumerate(dataset):
            self.assertEqual(waveform_spk1, self.samples[i][0])
            self.assertEqual(waveform_spk2, self.samples[i][1])
            assert sample_rate == self.samples[i][2]
            assert label == self.samples[i][3]
            assert file_id_spk1 == self.samples[i][4]
            assert file_id_spk2 == self.samples[i][5]
            num_samples += 1

        assert num_samples == len(self.samples)
