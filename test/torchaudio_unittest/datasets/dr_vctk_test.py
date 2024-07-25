from pathlib import Path

import pytest
from torchaudio.datasets import dr_vctk
from torchaudio_unittest.common_utils import get_whitenoise, save_wav, TempDirMixin, TorchaudioTestCase


_SUBSETS = ["train", "test"]
_CONDITIONS = ["clean", "device-recorded"]
_SOURCES = ["DR-VCTK_Office1_ClosedWindow", "DR-VCTK_Office1_OpenedWindow"]
_SPEAKER_IDS = range(226, 230)
_CHANNEL_IDS = range(1, 6)


def get_mock_dataset(root_dir):
    """
    root_dir: root directory of the mocked data
    """
    mocked_samples = {}

    dataset_dir = Path(root_dir) / "DR-VCTK" / "DR-VCTK"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config_dir = dataset_dir / "configurations"
    config_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    seed = 0

    for subset in _SUBSETS:
        mocked_samples[subset] = []

        for condition in _CONDITIONS:
            audio_dir = dataset_dir / f"{condition}_{subset}set_wav_16k"
            audio_dir.mkdir(parents=True, exist_ok=True)

        config_filepath = config_dir / f"{subset}_ch_log.txt"
        with open(config_filepath, "w") as f:
            if subset == "train":
                f.write("\n")
            f.write("File Name\tMain Source\tChannel Idx\n")

            for speaker_id in _SPEAKER_IDS:
                utterance_id = 1
                for source in _SOURCES:
                    for channel_id in _CHANNEL_IDS:
                        filename = f"p{speaker_id}_{utterance_id:03d}.wav"
                        f.write(f"{filename}\t{source}\t{channel_id}\n")

                        data = {}
                        for condition in _CONDITIONS:
                            data[condition] = get_whitenoise(
                                sample_rate=sample_rate, duration=0.01, n_channels=1, dtype="float32", seed=seed
                            )
                            audio_dir = dataset_dir / f"{condition}_{subset}set_wav_16k"
                            audio_file_path = audio_dir / filename
                            save_wav(audio_file_path, data[condition], sample_rate)
                            seed += 1

                        sample = (
                            data[_CONDITIONS[0]],
                            sample_rate,
                            data[_CONDITIONS[1]],
                            sample_rate,
                            "p" + str(speaker_id),
                            f"{utterance_id:03d}",
                            source,
                            channel_id,
                        )
                        mocked_samples[subset].append(sample)
                        utterance_id += 1

    return mocked_samples


class TestDRVCTK(TempDirMixin, TorchaudioTestCase):

    root_dir = None
    samples = {}

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = get_mock_dataset(cls.root_dir)

    def _test_dr_vctk(self, dataset, subset):
        num_samples = 0
        for i, (
            waveform_clean,
            sample_rate_clean,
            waveform_dr,
            sample_rate_dr,
            speaker_id,
            utterance_id,
            source,
            channel_id,
        ) in enumerate(dataset):
            self.assertEqual(waveform_clean, self.samples[subset][i][0], atol=5e-5, rtol=1e-8)
            assert sample_rate_clean == self.samples[subset][i][1]
            self.assertEqual(waveform_dr, self.samples[subset][i][2], atol=5e-5, rtol=1e-8)
            assert sample_rate_dr == self.samples[subset][i][3]
            assert speaker_id == self.samples[subset][i][4]
            assert utterance_id == self.samples[subset][i][5]
            assert source == self.samples[subset][i][6]
            assert channel_id == self.samples[subset][i][7]

            num_samples += 1

        assert num_samples == len(self.samples[subset])

    def test_dr_vctk_train_str(self):
        subset = "train"
        dataset = dr_vctk.DR_VCTK(self.root_dir, subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_test_str(self):
        subset = "test"
        dataset = dr_vctk.DR_VCTK(self.root_dir, subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_train_path(self):
        subset = "train"
        dataset = dr_vctk.DR_VCTK(Path(self.root_dir), subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_test_path(self):
        subset = "test"
        dataset = dr_vctk.DR_VCTK(Path(self.root_dir), subset=subset)
        self._test_dr_vctk(dataset, subset)

    def test_dr_vctk_invalid_subset(self):
        subset = "invalid"
        with pytest.raises(RuntimeError, match=f"The subset '{subset}' does not match any of the supported subsets"):
            dr_vctk.DR_VCTK(self.root_dir, subset=subset)
