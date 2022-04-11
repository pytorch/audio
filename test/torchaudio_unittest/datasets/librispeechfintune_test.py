import os

from torchaudio.datasets import librispeech_finetune
from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
)


# Used to generate a unique transcript for each dummy audio file
_NUMBERS = ["ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE"]


def _save_sample(dataset_dir):
    mocked_data = []
    sample_rate = 16000  # 16kHz
    seed = 0
    for subset in ["clean", "other"]:
        subset_dir = os.path.join(dataset_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)

        for speaker_id in range(5):
            speaker_path = os.path.join(subset_dir, str(speaker_id))
            os.makedirs(speaker_path, exist_ok=True)

            for chapter_id in range(3):
                chapter_path = os.path.join(speaker_path, str(chapter_id))
                os.makedirs(chapter_path, exist_ok=True)
                trans_content = []

                for utterance_id in range(3):
                    filename = f"{speaker_id}-{chapter_id}-{utterance_id:04d}.flac"
                    path = os.path.join(chapter_path, filename)

                    transcript = " ".join([_NUMBERS[x] for x in [speaker_id, chapter_id, utterance_id]])
                    trans_content.append(f"{speaker_id}-{chapter_id}-{utterance_id:04d} {transcript}")

                    data = get_whitenoise(
                        sample_rate=sample_rate, duration=0.01, n_channels=1, dtype="float32", seed=seed
                    )
                    save_wav(path, data, sample_rate)
                    print(path)
                    sample = (data, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
                    mocked_data.append(sample)

                    seed += 1

                trans_filename = f"{speaker_id}-{chapter_id}.trans.txt"
                trans_path = os.path.join(chapter_path, trans_filename)
                with open(trans_path, "w") as f:
                    f.write("\n".join(trans_content))
    return mocked_data


def get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    mocked_data_10min, mocked_data_1h, mocked_data_10h = [], [], []
    dataset_dir = os.path.join(root_dir, "librispeech_finetuning", "1h", "0")
    os.makedirs(dataset_dir, exist_ok=True)
    mocked_data_10min = _save_sample(dataset_dir)
    mocked_data_1h += mocked_data_10min
    for i in range(1, 6):
        dataset_dir = os.path.join(root_dir, "librispeech_finetuning", "1h", str(i))
        os.makedirs(dataset_dir, exist_ok=True)
        mocked_data_1h += _save_sample(dataset_dir)
    mocked_data_10h += mocked_data_1h

    dataset_dir = os.path.join(root_dir, "librispeech_finetuning", "9h")
    os.makedirs(dataset_dir, exist_ok=True)
    mocked_data_10h += _save_sample(dataset_dir)

    return mocked_data_10min, mocked_data_1h, mocked_data_10h


class TestLibriSpeechFineTune(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    samples_10min = []
    samples_1h = []
    samples_10h = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        (cls.samples_10min, cls.samples_1h, cls.samples_10h) = get_mock_dataset(cls.root_dir)

    def _test_librispeech(self, dataset, samples):
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

    def test_librispeech_10min(self):
        dataset = librispeech_finetune.LibriSpeechFineTune(self.root_dir, split="10min")
        self._test_librispeech(dataset, self.samples_10min)

    def test_librispeech_1h(self):
        dataset = librispeech_finetune.LibriSpeechFineTune(self.root_dir, split="1h")
        self._test_librispeech(dataset, self.samples_1h)

    def test_librispeech_10h(self):
        dataset = librispeech_finetune.LibriSpeechFineTune(self.root_dir, split="10h")
        self._test_librispeech(dataset, self.samples_10h)
