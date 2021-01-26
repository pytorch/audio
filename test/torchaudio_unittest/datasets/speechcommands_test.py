import os
from pathlib import Path

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    normalize_wav,
    save_wav,
)

from torchaudio.datasets import speechcommands

_LABELS = [
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
]


def get_mock_dataset(dataset_dir):
    """
    dataset_dir: directory to the mocked dataset
    """
    mocked_samples = []
    mocked_train_samples = []
    mocked_valid_samples = []
    mocked_test_samples = []
    os.makedirs(dataset_dir, exist_ok=True)
    sample_rate = 16000  # 16kHz sample rate
    seed = 0
    valid_file = os.path.join(dataset_dir, "validation_list.txt")
    test_file = os.path.join(dataset_dir, "testing_list.txt")
    with open(valid_file, "w") as valid, open(test_file, "w") as test:
        for label in _LABELS:
            path = os.path.join(dataset_dir, label)
            os.makedirs(path, exist_ok=True)
            for j in range(6):
                # generate hash ID for speaker
                speaker = "{:08x}".format(j)

                for utterance in range(3):
                    filename = f"{speaker}{speechcommands.HASH_DIVIDER}{utterance}.wav"
                    file_path = os.path.join(path, filename)
                    seed += 1
                    data = get_whitenoise(
                        sample_rate=sample_rate,
                        duration=0.01,
                        n_channels=1,
                        dtype="int16",
                        seed=seed,
                    )
                    save_wav(file_path, data, sample_rate)
                    sample = (
                        normalize_wav(data),
                        sample_rate,
                        label,
                        speaker,
                        utterance,
                    )
                    mocked_samples.append(sample)
                    if j < 2:
                        mocked_train_samples.append(sample)
                    elif j < 4:
                        valid.write(f'{label}/{filename}\n')
                        mocked_valid_samples.append(sample)
                    elif j < 6:
                        test.write(f'{label}/{filename}\n')
                        mocked_test_samples.append(sample)
    return mocked_samples, mocked_train_samples, mocked_valid_samples, mocked_test_samples


class TestSpeechCommands(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    samples = []
    train_samples = []
    valid_samples = []
    test_samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(
            cls.root_dir, speechcommands.FOLDER_IN_ARCHIVE, speechcommands.URL
        )
        cls.samples, cls.train_samples, cls.valid_samples, cls.test_samples = get_mock_dataset(dataset_dir)

    def _testSpeechCommands(self, dataset, data_samples):
        num_samples = 0
        for i, (data, sample_rate, label, speaker_id, utterance_number) in enumerate(
                dataset
        ):
            self.assertEqual(data, data_samples[i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate == data_samples[i][1]
            assert label == data_samples[i][2]
            assert speaker_id == data_samples[i][3]
            assert utterance_number == data_samples[i][4]
            num_samples += 1

        assert num_samples == len(data_samples)

    def testSpeechCommands_str(self):
        dataset = speechcommands.SPEECHCOMMANDS(self.root_dir)
        self._testSpeechCommands(dataset, self.samples)

    def testSpeechCommands_path(self):
        dataset = speechcommands.SPEECHCOMMANDS(Path(self.root_dir))
        self._testSpeechCommands(dataset, self.samples)

    def testSpeechCommandsSubsetTrain(self):
        dataset = speechcommands.SPEECHCOMMANDS(self.root_dir, subset="training")
        self._testSpeechCommands(dataset, self.train_samples)

    def testSpeechCommandsSubsetValid(self):
        dataset = speechcommands.SPEECHCOMMANDS(self.root_dir, subset="validation")
        self._testSpeechCommands(dataset, self.valid_samples)

    def testSpeechCommandsSubsetTest(self):
        dataset = speechcommands.SPEECHCOMMANDS(self.root_dir, subset="testing")
        self._testSpeechCommands(dataset, self.test_samples)

    def testSpeechCommandsSum(self):
        dataset_all = speechcommands.SPEECHCOMMANDS(self.root_dir)
        dataset_train = speechcommands.SPEECHCOMMANDS(self.root_dir, subset="training")
        dataset_valid = speechcommands.SPEECHCOMMANDS(self.root_dir, subset="validation")
        dataset_test = speechcommands.SPEECHCOMMANDS(self.root_dir, subset="testing")

        assert len(dataset_train) + len(dataset_valid) + len(dataset_test) == len(dataset_all)
