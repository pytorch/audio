import os
from collections import defaultdict
from pathlib import Path

from parameterized import parameterized
from torchaudio.datasets import quesst14
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase


def _get_filename(folder, index):
    if folder == "Audio":
        return f"quesst14_{index:05d}.wav"
    elif folder == "dev_queries":
        return f"quesst14_dev_{index:04d}.wav"
    elif folder == "eval_queries":
        return f"quesst14_eval_{index:04d}.wav"
    return


def _get_key(folder):
    folder_key_mapping = {
        "Audio": "utterances",
        "dev_queries": "dev",
        "eval_queries": "eval",
    }
    return folder_key_mapping[folder]


def _save_sample(dataset_dir, folder, language, index, sample_rate, seed):
    # create and save audio samples to corresponding files
    path = os.path.join(dataset_dir, folder)
    os.makedirs(path, exist_ok=True)
    filename = _get_filename(folder, index)
    file_path = os.path.join(path, filename)

    data = get_whitenoise(
        sample_rate=sample_rate,
        duration=0.01,
        n_channels=1,
        seed=seed,
    )
    save_wav(file_path, data, sample_rate)

    sample = (data, sample_rate, Path(file_path).with_suffix("").name)

    # add audio files and language data to language key files
    scoring_path = os.path.join(dataset_dir, "scoring")
    os.makedirs(scoring_path, exist_ok=True)
    wav_file = f"quesst14Database/{folder}/{filename}"
    line = f"{wav_file} {language}"

    key = _get_key(folder)
    language_key_file = f"language_key_{key}.lst"
    language_key_file = os.path.join(scoring_path, language_key_file)
    with open(language_key_file, "a") as f:
        f.write(line + "\n")

    return sample


def _get_mocked_samples(dataset_dir, folder, sample_rate, seed):
    samples_per_language = 2

    samples_map = defaultdict(list)
    samples_all = []

    curr_idx = 0
    for language in quesst14._LANGUAGES:
        for _ in range(samples_per_language):
            sample = _save_sample(dataset_dir, folder, language, curr_idx, sample_rate, seed)
            samples_map[language].append(sample)
            samples_all.append(sample)

            curr_idx += 1
    return samples_map, samples_all


def get_mock_dataset(dataset_dir):
    """
    dataset_dir: directory to the mocked dataset
    """
    os.makedirs(dataset_dir, exist_ok=True)
    sample_rate = 8000

    audio_seed = 0
    dev_seed = 1
    eval_seed = 2

    mocked_utterances, mocked_utterances_all = _get_mocked_samples(dataset_dir, "Audio", sample_rate, audio_seed)
    mocked_dev_samples, mocked_dev_samples_all = _get_mocked_samples(dataset_dir, "dev_queries", sample_rate, dev_seed)
    mocked_eval_samples, mocked_eval_samples_all = _get_mocked_samples(
        dataset_dir, "eval_queries", sample_rate, eval_seed
    )

    return (
        mocked_utterances,
        mocked_dev_samples,
        mocked_eval_samples,
        mocked_utterances_all,
        mocked_dev_samples_all,
        mocked_eval_samples_all,
    )


class TestQuesst14(TempDirMixin, TorchaudioTestCase):
    root_dir = None

    utterances = {}
    dev_samples = {}
    eval_samples = {}
    utterances_all = []
    dev_samples_all = []
    eval_samples_all = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        dataset_dir = os.path.join(cls.root_dir, "quesst14Database")
        (
            cls.utterances,
            cls.dev_samples,
            cls.eval_samples,
            cls.utterances_all,
            cls.dev_samples_all,
            cls.eval_samples_all,
        ) = get_mock_dataset(dataset_dir)

    def _testQuesst14(self, dataset, data_samples):
        num_samples = 0
        for i, (data, sample_rate, name) in enumerate(dataset):
            self.assertEqual(data, data_samples[i][0])
            assert sample_rate == data_samples[i][1]
            assert name == data_samples[i][2]
            num_samples += 1

        assert num_samples == len(data_samples)

    def testQuesst14SubsetDocs(self):
        dataset = quesst14.QUESST14(self.root_dir, language=None, subset="docs")
        self._testQuesst14(dataset, self.utterances_all)

    def testQuesst14SubsetDev(self):
        dataset = quesst14.QUESST14(self.root_dir, language=None, subset="dev")
        self._testQuesst14(dataset, self.dev_samples_all)

    def testQuesst14SubsetEval(self):
        dataset = quesst14.QUESST14(self.root_dir, language=None, subset="eval")
        self._testQuesst14(dataset, self.eval_samples_all)

    @parameterized.expand(quesst14._LANGUAGES)
    def testQuesst14DocsSingleLanguage(self, language):
        dataset = quesst14.QUESST14(self.root_dir, language=language, subset="docs")
        self._testQuesst14(dataset, self.utterances[language])

    @parameterized.expand(quesst14._LANGUAGES)
    def testQuesst14DevSingleLanguage(self, language):
        dataset = quesst14.QUESST14(self.root_dir, language=language, subset="dev")
        self._testQuesst14(dataset, self.dev_samples[language])

    @parameterized.expand(quesst14._LANGUAGES)
    def testQuesst14EvalSingleLanguage(self, language):
        dataset = quesst14.QUESST14(self.root_dir, language=language, subset="eval")
        self._testQuesst14(dataset, self.eval_samples[language])
