import csv
import os

from torchaudio.datasets import ljspeech

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    normalize_wav,
    save_wav,
)


class TestLJSpeech(TempDirMixin, TorchaudioTestCase):
    backend = "default"

    root_dir = None
    data = []
    transcripts = [
        "Test transcript 1",
        "Test transcript 2",
        "Test transcript 3",
        "In 1465 Sweynheim and Pannartz began printing in the monastery of Subiaco near Rome,"
    ]

    normalized_transcripts = [
        "Test transcript one",
        "Test transcript two",
        "Test transcript three",
        "In fourteen sixty-five Sweynheim and Pannartz began printing in the monastery of Subiaco near Rome,"
    ]

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        base_dir = os.path.join(cls.root_dir, "LJSpeech-1.1")
        archive_dir = os.path.join(base_dir, "wavs")
        os.makedirs(archive_dir, exist_ok=True)
        metadata_path = os.path.join(base_dir, "metadata.csv")
        sample_rate = 22050

        with open(metadata_path, mode="w", newline='') as metadata_file:
            metadata_writer = csv.writer(
                metadata_file, delimiter="|", quoting=csv.QUOTE_NONE
            )
            for i, (transcript, normalized_transcript) in enumerate(
                zip(cls.transcripts, cls.normalized_transcripts)
            ):
                fileid = f'LJ001-{i:04d}'
                metadata_writer.writerow([fileid, transcript, normalized_transcript])
                filename = fileid + ".wav"
                path = os.path.join(archive_dir, filename)
                data = get_whitenoise(
                    sample_rate=sample_rate, duration=1, n_channels=1, dtype="int16", seed=i
                )
                save_wav(path, data, sample_rate)
                cls.data.append(normalize_wav(data))

    def test_ljspeech(self):
        dataset = ljspeech.LJSPEECH(self.root_dir)
        n_ite = 0
        for i, (waveform, sample_rate, transcript, normalized_transcript) in enumerate(
            dataset
        ):
            expected_transcript = self.transcripts[i]
            expected_normalized_transcript = self.normalized_transcripts[i]
            expected_data = self.data[i]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == sample_rate
            assert transcript == expected_transcript
            assert normalized_transcript == expected_normalized_transcript
            n_ite += 1
        assert n_ite == len(self.data)
