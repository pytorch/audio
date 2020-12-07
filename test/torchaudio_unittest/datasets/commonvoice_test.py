import os
import csv
import tarfile
from pathlib import Path

from torchaudio.datasets import commonvoice
from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    get_whitenoise,
    save_wav,
    normalize_wav,
)

_HEADERS = [
    "client_ids",
    "path",
    "sentence",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accent",
]

# Note: extension is changed to wav for the sake of test
# Note: the first content is missing values for `age`, `gender` and `accent` as in the original data.
_TRAIN_CSV_CONTENTS = [
    [
        "9d16c5d980247861130e0480e2719f448be73d86a496c36d01a477cbdecd8cfd1399403d7a77bf458d211a70711b2da0845c",
        "common_voice_en_18885784.wav",
        "He was accorded a State funeral, and was buried in Drayton and Toowoomba Cemetery.",
        "2",
        "0",
        "",
        "",
        ""
    ],
    [
        "c82eb9291328620f06025a1f8112b909099e447e485e99236cb87df008650250e79fea5ca772061fb6a370830847b9c44d20",
        "common_voice_en_556542.wav",
        "Once more into the breach",
        "2",
        "0",
        "thirties",
        "male",
        "us",
    ],
    [
        "f74d880c5ad4c5917f314a604d3fc4805159d255796fb9f8defca35333ecc002bdf53dc463503c12674ea840b21b4a507b7c",
        "common_voice_en_18607573.wav",
        "Caddy, show Miss Clare and Miss Summerson their rooms.",
        "2",
        "0",
        "twenties",
        "male",
        "canada",
    ],
]


def _make_dataset(root_dir, sample_rate=48000):
    # The path convention commonvoice uses
    base_dir = os.path.join(root_dir, "CommonVoice", "cv-corpus-4-2019-12-10", "en")
    audio_dir = os.path.join(base_dir, "clips")
    tsv_path = os.path.join(base_dir, "train.tsv")

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    # Tsv file name difference does not mean different subset, testing as a whole dataset here
    print(tsv_path)
    with open(tsv_path, "w", newline='') as tsv:
        writer = csv.writer(tsv, delimiter='\t')
        writer.writerow(_HEADERS)
        for content in _TRAIN_CSV_CONTENTS:
            writer.writerow(content)

    # Generate audio files
    expected = []
    for i, content in enumerate(_TRAIN_CSV_CONTENTS):
        audio_path = os.path.join(audio_dir, content[1])
        data = get_whitenoise(
            sample_rate=sample_rate, duration=1, n_channels=1, seed=i, dtype='float32')
        save_wav(audio_path, data, sample_rate)
        print(audio_path)
        expected.append((normalize_wav(data), sample_rate, dict(zip(_HEADERS, content))))
    return expected


def _make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


class TestCommonVoice(TempDirMixin, TorchaudioTestCase):
    backend = 'default'

    root_dir = ""
    expected = []

    @classmethod
    def setUpClass(cls):
        root_dir = cls.get_base_temp_dir()
        tmp_dir = os.path.join(root_dir, 'tmp')
        expected = _make_dataset(tmp_dir)
        source_dir = os.path.join(tmp_dir, 'CommonVoice')
        arch_path = os.path.join(root_dir, 'en.tar.gz')
        _make_tarfile(arch_path, source_dir)

        cls.root_dir = root_dir
        cls.expected = expected

    def _test_commonvoice(self, dataset):
        n_ite = 0
        for i, (waveform, sample_rate, dictionary) in enumerate(dataset):
            expected_dictionary = self.expected[i][2]
            expected_data = self.expected[i][0]
            self.assertEqual(expected_data, waveform, atol=5e-5, rtol=1e-8)
            assert sample_rate == 48000
            assert dictionary == expected_dictionary
            n_ite += 1
        assert n_ite == len(self.expected)

    def test_commonvoice_str(self):
        dataset = commonvoice.COMMONVOICE(self.root_dir)
        self._test_commonvoice(dataset)

    def test_commonvoice_path(self):
        dataset = commonvoice.COMMONVOICE(Path(self.root_dir))
        self._test_commonvoice(dataset)
