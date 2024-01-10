import os

from torchaudio.datasets import librilight_limited
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase


# Used to generate a unique transcript for each dummy audio file
_NUMBERS = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]


def _save_sample(file_path, speaker_id, chapter_id, utterance_id, sample_rate, seed):
    filename = f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
    path = os.path.join(file_path, filename)
    data = get_whitenoise(sample_rate=sample_rate, duration=0.01, n_channels=1, dtype="float32", seed=seed)
    transcript = " ".join([_NUMBERS[x] for x in [speaker_id, chapter_id, utterance_id]])
    save_wav(path, data, sample_rate)
    sample = (data, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
    return sample


def get_mock_dataset(dataset_dir: str):
    """Create mocked dataset for a sub directory.

    Args:
        dataset_dir (str): the path of the sub directory.
        The structure is: audio_type/speaker_id/chapter_id/filename.flac
    """
    mocked_data = []
    sample_rate = 16000  # 16kHz
    seed = 0
    for audio_type in ["clean", "other"]:
        for speaker_id in range(5):
            for chapter_id in range(3):
                file_path = os.path.join(dataset_dir, audio_type, str(speaker_id), str(chapter_id))
                os.makedirs(file_path, exist_ok=True)
                trans_content = []
                for utterance_id in range(3):
                    sample = _save_sample(file_path, speaker_id, chapter_id, utterance_id, sample_rate, seed)
                    trans_content.append(f"{sample[3]}-{sample[4]}-{sample[5]:04d} {sample[2]}")
                    mocked_data.append(sample)
                    seed += 1
                trans_filename = f"{speaker_id}-{chapter_id}.trans.txt"
                trans_path = os.path.join(file_path, trans_filename)
                with open(trans_path, "w") as f:
                    f.write("\n".join(trans_content))
    return mocked_data


def get_mock_datasets(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    mocked_data_10min, mocked_data_1h, mocked_data_10h = [], [], []
    dataset_dir = os.path.join(root_dir, "librispeech_finetuning", "1h", "0")
    os.makedirs(dataset_dir, exist_ok=True)
    mocked_data_10min = get_mock_dataset(dataset_dir)
    mocked_data_1h += mocked_data_10min
    for i in range(1, 6):
        dataset_dir = os.path.join(root_dir, "librispeech_finetuning", "1h", str(i))
        os.makedirs(dataset_dir, exist_ok=True)
        mocked_data_1h += get_mock_dataset(dataset_dir)
    mocked_data_10h += mocked_data_1h

    dataset_dir = os.path.join(root_dir, "librispeech_finetuning", "9h")
    os.makedirs(dataset_dir, exist_ok=True)
    mocked_data_10h += get_mock_dataset(dataset_dir)

    return mocked_data_10min, mocked_data_1h, mocked_data_10h


class TestLibriLightLimited(TempDirMixin, TorchaudioTestCase):

    root_dir = None
    samples_10min = []
    samples_1h = []
    samples_10h = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        (cls.samples_10min, cls.samples_1h, cls.samples_10h) = get_mock_datasets(cls.root_dir)

    def _test_librilightlimited(self, dataset, samples):
        num_samples = 0
        for i, (data, sample_rate, transcript, speaker_id, chapter_id, utterance_id) in enumerate(dataset):
            self.assertEqual(data, samples[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == samples[i][1]
            assert transcript == samples[i][2]
            assert speaker_id == samples[i][3]
            assert chapter_id == samples[i][4]
            assert utterance_id == samples[i][5]
            num_samples += 1

        assert num_samples == len(samples)

    def test_librilightlimited_10min(self):
        dataset = librilight_limited.LibriLightLimited(self.root_dir, subset="10min")
        self._test_librilightlimited(dataset, self.samples_10min)

    def test_librilightlimited_1h(self):
        dataset = librilight_limited.LibriLightLimited(self.root_dir, subset="1h")
        self._test_librilightlimited(dataset, self.samples_1h)

    def test_librilightlimited_10h(self):
        dataset = librilight_limited.LibriLightLimited(self.root_dir, subset="10h")
        self._test_librilightlimited(dataset, self.samples_10h)
