import csv
import os
from pathlib import Path

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    normalize_wav,
    save_wav,
)

from torchaudio.datasets import ljspeech

_TRANSCRIPTS = [
    "Test transcript 1",
    "Test transcript 2",
    "Test transcript 3",
    "In 1465 Sweynheim and Pannartz began printing in the monastery of Subiaco near Rome,"
]

_NORMALIZED_TRANSCRIPT = [
    "Test transcript one",
    "Test transcript two",
    "Test transcript three",
    "In fourteen sixty-five Sweynheim and Pannartz began printing in the monastery of Subiaco near Rome,"
]


def get_mock_dataset(root_dir):
    """
    root_dir: path to the mocked dataset
    """
    mocked_data = []
    base_dir = os.path.join(root_dir, "LJSpeech-1.1")
    archive_dir = os.path.join(base_dir, "wavs")
    os.makedirs(archive_dir, exist_ok=True)
    metadata_path = os.path.join(base_dir, "metadata.csv")
    sample_rate = 22050

    with open(metadata_path, mode="w", newline='') as metadata_file:
        metadata_writer = csv.writer(
            metadata_file, delimiter="|", quoting=csv.QUOTE_NONE
        )
        for i, (transcript, normalized_transcript) in enumerate(
                zip(_TRANSCRIPTS, _NORMALIZED_TRANSCRIPT)
        ):
            fileid = f'LJ001-{i:04d}'
            metadata_writer.writerow([fileid, transcript, normalized_transcript])
            filename = fileid + ".wav"
            path = os.path.join(archive_dir, filename)
            data = get_whitenoise(
                sample_rate=sample_rate, duration=1, n_channels=1, dtype="int16", seed=i
            )
            save_wav(path, data, sample_rate)
            mocked_data.append(normalize_wav(data))
    return mocked_data


class TestLJSpeech(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    data = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.data = get_mock_dataset(cls.root_dir)

    def _test_ljspeech(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, transcript, normalized_transcript) in enumerate(
                dataset
        ):
            expected_transcript = _TRANSCRIPTS[i]
            expected_normalized_transcript = _NORMALIZED_TRANSCRIPT[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == sample_rate
            assert transcript == expected_transcript
            assert normalized_transcript == expected_normalized_transcript
            n_ite += 1
        assert n_ite == len(self.data)

    def test_ljspeech_str(self):
        dataset = ljspeech.LJSPEECH(self.root_dir)
        self._test_ljspeech(dataset)

    def test_ljspeech_path(self):
        dataset = ljspeech.LJSPEECH(Path(self.root_dir))
        self._test_ljspeech(dataset)
