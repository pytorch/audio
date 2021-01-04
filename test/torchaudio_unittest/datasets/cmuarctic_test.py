import os
from pathlib import Path

from torchaudio.datasets import cmuarctic

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)

_UTTERANCE = "This is a test utterance."


def get_mock_dataset(root_dir):
    """
    prepares mocked dataset
    """
    mocked_data = []
    sample_rate = 16000
    base_dir = os.path.join(root_dir, "ARCTIC", "cmu_us_aew_arctic")
    txt_dir = os.path.join(base_dir, "etc")
    os.makedirs(txt_dir, exist_ok=True)
    txt_file = os.path.join(txt_dir, "txt.done.data")
    audio_dir = os.path.join(base_dir, "wav")
    os.makedirs(audio_dir, exist_ok=True)

    seed = 42
    with open(txt_file, "w") as txt:
        for c in ["a", "b"]:
            for i in range(5):
                utterance_id = f"arctic_{c}{i:04d}"
                path = os.path.join(audio_dir, f"{utterance_id}.wav")
                data = get_whitenoise(
                    sample_rate=sample_rate,
                    duration=3,
                    n_channels=1,
                    dtype="int16",
                    seed=seed,
                )
                save_wav(path, data, sample_rate)
                sample = (
                    normalize_wav(data),
                    sample_rate,
                    _UTTERANCE,
                    utterance_id.split("_")[1],
                )
                mocked_data.append(sample)
                txt.write(f'( {utterance_id} "{_UTTERANCE}" )\n')
                seed += 1
    return mocked_data


class TestCMUARCTIC(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = get_mock_dataset(cls.root_dir)

    def _test_cmuarctic(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, utterance, utterance_id) in enumerate(dataset):
            expected_sample = self.samples[i]
            assert sample_rate == expected_sample[1]
            assert utterance == expected_sample[2]
            assert utterance_id == expected_sample[3]
            self.assertEqual(expected_sample[0], waveform, atol=5e-5, rtol=1e-8)
            n_ite += 1
        assert n_ite == len(self.samples)

    def test_cmuarctic_str(self):
        dataset = cmuarctic.CMUARCTIC(self.root_dir)
        self._test_cmuarctic(dataset)

    def test_cmuarctic_path(self):
        dataset = cmuarctic.CMUARCTIC(Path(self.root_dir))
        self._test_cmuarctic(dataset)
