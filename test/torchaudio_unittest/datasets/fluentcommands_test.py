import csv
import os
import random
import string

from torchaudio.datasets import fluentcommands
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase

HEADER = ["", "path", "speakerId", "transcription", "action", "object", "location"]
SLOTS = ["action", "object", "location"]
ACTIONS = ["activate", "deactivate"]
OBJECTS = ["lights", "volume"]
LOCATIONS = ["none", "kitchen", "bedroom"]
NUM_SPEAKERS = 5
SAMPLES_PER_SPEAKER = 10
SAMPLE_RATE = 16000


def _gen_rand_str(n: int, seed: int):
    random.seed(seed)
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _gen_csv(dataset_dir: str, subset: str, init_seed: int):
    data = []
    data.append(HEADER)

    idx = 0
    seed = init_seed
    for _ in range(NUM_SPEAKERS):
        speaker_id = _gen_rand_str(5, seed=seed)
        speaker_dir = os.path.join(dataset_dir, "wavs", "speakers", speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        for _ in range(SAMPLES_PER_SPEAKER):
            seed += 1
            filename = _gen_rand_str(10, seed=seed)
            path = f"wavs/speakers/{speaker_id}/{filename}.wav"

            random.seed(seed)
            transcription = ""
            act = random.choice(ACTIONS)
            obj = random.choice(OBJECTS)
            loc = random.choice(LOCATIONS)

            data.append([idx, path, speaker_id, transcription, act, obj, loc])

            idx += 1

    csv_path = os.path.join(dataset_dir, "data", f"{subset}_data.csv")
    with open(csv_path, "w", newline="") as csv_file:
        file_writer = csv.writer(csv_file)
        file_writer.writerows(data)

    return data


def _save_samples(dataset_dir: str, subset: str, seed: int):
    # generate csv file
    data = _gen_csv(dataset_dir, subset, seed)

    # iterate through csv file, save wavs to corresponding files
    header = data[0]
    data = data[1:]  # remove header
    path_idx = header.index("path")

    samples = []
    for row in data:
        wav = get_whitenoise(
            sample_rate=SAMPLE_RATE,
            duration=0.01,
            n_channels=1,
            seed=seed,
        )
        path = row[path_idx]
        filename = path.split("/")[-1]
        filename = filename.split(".")[0]
        speaker_id, transcription, act, obj, loc = row[2:]

        wav_file = os.path.join(dataset_dir, "wavs", "speakers", speaker_id, f"{filename}.wav")
        save_wav(wav_file, wav, SAMPLE_RATE)

        sample = wav, SAMPLE_RATE, filename, speaker_id, transcription, act, obj, loc
        samples.append(sample)

        seed += 1

    return samples


def get_mock_dataset(dataset_dir: str):
    data_folder = os.path.join(dataset_dir, "data")
    wav_folder = os.path.join(dataset_dir, "wavs", "speakers")

    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(wav_folder, exist_ok=True)

    mocked_train_samples = _save_samples(dataset_dir, "train", 1)
    mocked_valid_samples = _save_samples(dataset_dir, "valid", 111)
    mocked_test_samples = _save_samples(dataset_dir, "test", 1111)

    return mocked_train_samples, mocked_valid_samples, mocked_test_samples


class TestFluentSpeechCommands(TempDirMixin, TorchaudioTestCase):
    root_dir = None

    mocked_train_samples = []
    mocked_valid_samples = []
    mocked_test_samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(cls.root_dir, "fluent_speech_commands_dataset")
        (
            cls.mocked_train_samples,
            cls.mocked_valid_samples,
            cls.mocked_test_samples,
        ) = get_mock_dataset(dataset_dir)

    def _testFluentCommands(self, dataset, samples):
        num_samples = 0
        for i, data in enumerate(dataset):
            self.assertEqual(data, samples[i])
            num_samples += 1

        assert num_samples == len(samples)

    def testFluentCommandsTrain(self):
        dataset = fluentcommands.FluentSpeechCommands(self.root_dir, subset="train")
        self._testFluentCommands(dataset, self.mocked_train_samples)

    def testFluentCommandsValid(self):
        dataset = fluentcommands.FluentSpeechCommands(self.root_dir, subset="valid")
        self._testFluentCommands(dataset, self.mocked_valid_samples)

    def testFluentCommandsTest(self):
        dataset = fluentcommands.FluentSpeechCommands(self.root_dir, subset="test")
        self._testFluentCommands(dataset, self.mocked_test_samples)
