import os

from parameterized import parameterized

from torchaudio.datasets import LibriMix
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase

_SAMPLE_RATE = 8000
_TASKS_TO_MIXTURE = {
    "sep_clean": "mix_clean",
    "enh_single": "mix_single",
    "enh_both": "mix_both",
    "sep_noisy": "mix_both",
}


def _save_wav(filepath: str, seed: int):
    wav = get_whitenoise(
        sample_rate=_SAMPLE_RATE,
        duration=0.01,
        n_channels=1,
        seed=seed,
    )
    save_wav(filepath, wav, _SAMPLE_RATE)
    return wav


def get_mock_dataset(root_dir: str, num_speaker: int):
    """
    root_dir: directory to the mocked dataset
    """
    mocked_data = []
    seed = 0
    base_dir = os.path.join(root_dir, f"Libri{num_speaker}Mix", "wav8k", "min", "train-360")
    os.makedirs(base_dir, exist_ok=True)
    for utterance_id in range(10):
        filename = f"{utterance_id}.wav"
        task_outputs = {}
        for task in _TASKS_TO_MIXTURE:
            # create mixture folder. The folder names depends on the task.
            mixture_folder = _TASKS_TO_MIXTURE[task]
            mixture_dir = os.path.join(base_dir, mixture_folder)
            os.makedirs(mixture_dir, exist_ok=True)
            mixture_path = os.path.join(mixture_dir, filename)
            mixture = _save_wav(mixture_path, seed)
            sources = []
            if task == "enh_both":
                sources = [task_outputs["sep_clean"][1]]
            else:
                for speaker_id in range(num_speaker):
                    source_dir = os.path.join(base_dir, f"s{speaker_id+1}")
                    os.makedirs(source_dir, exist_ok=True)
                    source_path = os.path.join(source_dir, filename)
                    if os.path.exists(source_path):
                        sources = task_outputs["sep_clean"][2]
                        break
                    else:
                        source = _save_wav(source_path, seed)
                        sources.append(source)
                        seed += 1
            task_outputs[task] = (_SAMPLE_RATE, mixture, sources)
        mocked_data.append(task_outputs)

    return mocked_data


class TestLibriMix(TempDirMixin, TorchaudioTestCase):

    root_dir = None
    samples_2spk = []
    samples_3spk = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples_2spk = get_mock_dataset(cls.root_dir, 2)
        cls.samples_3spk = get_mock_dataset(cls.root_dir, 3)

    def _test_librimix(self, dataset, samples, task):
        num_samples = 0
        for i, (sample_rate, mixture, sources) in enumerate(dataset):
            assert sample_rate == samples[i][task][0]
            self.assertEqual(mixture, samples[i][task][1])
            assert len(sources) == len(samples[i][task][2])
            for j in range(len(sources)):
                self.assertEqual(sources[j], samples[i][task][2][j])
            num_samples += 1

        assert num_samples == len(samples)

    @parameterized.expand([("sep_clean"), ("enh_single",), ("enh_both",), ("sep_noisy",)])
    def test_librimix_2speaker(self, task):
        dataset = LibriMix(self.root_dir, num_speakers=2, sample_rate=_SAMPLE_RATE, task=task)
        self._test_librimix(dataset, self.samples_2spk, task)

    @parameterized.expand([("sep_clean"), ("enh_single",), ("enh_both",), ("sep_noisy",)])
    def test_librimix_3speaker(self, task):
        dataset = LibriMix(self.root_dir, num_speakers=3, sample_rate=_SAMPLE_RATE, task=task)
        self._test_librimix(dataset, self.samples_3spk, task)
