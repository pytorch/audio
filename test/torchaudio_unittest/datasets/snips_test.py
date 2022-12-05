import os

from torchaudio.datasets import snips
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase


_SAMPLE_RATE = 16000
_SPEAKERS = [
    "Aditi",
    "Amy",
    "Brian",
    "Emma",
    "Geraint",
    "Ivy",
    "Joanna",
    "Joey",
    "Justin",
    "Kendra",
    "Kimberly",
    "Matthew",
    "Nicole",
    "Raveena",
    "Russell",
    "Salli",
]


def _save_wav(filepath: str, seed: int):
    wav = get_whitenoise(
        sample_rate=_SAMPLE_RATE,
        duration=0.01,
        n_channels=1,
        seed=seed,
    )
    save_wav(filepath, wav, _SAMPLE_RATE)
    return wav


def _save_label(label_path: str, wav_stem: str, label: str):
    with open(label_path, "a") as f:
        f.write(f"{wav_stem} {label}\n")


def _get_mocked_samples(dataset_dir: str, subset: str, seed: int):
    samples = []
    subset_dir = os.path.join(dataset_dir, subset)
    label_path = os.path.join(dataset_dir, "all.iob.snips.txt")
    os.makedirs(subset_dir, exist_ok=True)
    num_utterance_per_split = 10
    for spk in _SPEAKERS:
        for i in range(num_utterance_per_split):
            wav_stem = f"{spk}-snips-{subset}-{i}"
            wav_path = os.path.join(subset_dir, f"{wav_stem}.wav")
            waveform = _save_wav(wav_path, seed)
            transcript, iob, intent = f"{spk}XXX", f"{spk}YYY", f"{spk}ZZZ"
            label = "BOS " + transcript + " EOS\tO " + iob + " " + intent
            _save_label(label_path, wav_stem, label)
            samples.append((waveform, _SAMPLE_RATE, wav_stem, transcript, iob, intent))
    return samples


def get_mock_datasets(dataset_dir):
    """
    dataset_dir: directory to the mocked dataset
    """
    os.makedirs(dataset_dir, exist_ok=True)

    train_seed = 0
    valid_seed = 1
    test_seed = 2

    mocked_train_samples = _get_mocked_samples(dataset_dir, "train", train_seed)
    mocked_valid_samples = _get_mocked_samples(dataset_dir, "valid", valid_seed)
    mocked_test_samples = _get_mocked_samples(dataset_dir, "test", test_seed)

    return (
        mocked_train_samples,
        mocked_valid_samples,
        mocked_test_samples,
    )


class TestSnips(TempDirMixin, TorchaudioTestCase):
    root_dir = None
    backend = "default"

    train_samples = {}
    valid_samples = {}
    test_samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(cls.root_dir, "SNIPS")
        (
            cls.train_samples,
            cls.valid_samples,
            cls.test_samples,
        ) = get_mock_datasets(dataset_dir)

    def _testSnips(self, dataset, data_samples):
        num_samples = 0
        for i, (data, sample_rate, file_name, transcript, iob, intent) in enumerate(dataset):
            self.assertEqual(data, data_samples[i][0])
            assert sample_rate == data_samples[i][1]
            assert file_name == data_samples[i][2]
            assert transcript == data_samples[i][3]
            assert iob == data_samples[i][4]
            assert intent == data_samples[i][5]
            num_samples += 1

        assert num_samples == len(data_samples)

    def testSnipsTrain(self):
        dataset = snips.Snips(self.root_dir, subset="train", audio_format="wav")
        self._testSnips(dataset, self.train_samples)

    def testSnipsValid(self):
        dataset = snips.Snips(self.root_dir, subset="valid", audio_format="wav")
        self._testSnips(dataset, self.valid_samples)

    def testSnipsTest(self):
        dataset = snips.Snips(self.root_dir, subset="test", audio_format="wav")
        self._testSnips(dataset, self.test_samples)
