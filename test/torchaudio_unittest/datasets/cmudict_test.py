import os
from pathlib import Path
import string

from torchaudio.datasets import CMUDict

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
)


def get_mock_dataset(root_dir):
    """
    root_dir: directory to the mocked dataset
    """
    mocked_dictionary = [
        ";;; # CMUdict  --  Major Version: 0.07",
        ";;; ",
        ";;; # $HeadURL$",
        "!EXCLAMATION-POINT  EH2 K S K L AH0 M EY1 SH AH0 N P OY2 N T",
        "\"CLOSE-QUOTE  K L OW1 Z K W OW1 T",
        "AALIYAH  AA2 L IY1 AA2",
        "AALSETH  AA1 L S EH0 TH",
        "AAMODT  AA1 M AH0 T",
        "AANCOR  AA1 N K AO2 R",
        "AARDEMA  AA0 R D EH1 M AH0",
        "AARDVARK  AA1 R D V AA2 R K",
    ]
    mocked_symbols = [
        "AA0", "AA1", "AA2", "AH0", "AO2", "D", "EH0", "EH2", "IY1", "K", "L", "M",
        "N", "P", "OW1", "OY2", "R", "S", "SH", "T", "TH", "W", "V", "Z",
    ]

    dict_file = os.path.join(root_dir, "cmudict-0.7b")
    symbol_file = os.path.join(root_dir, "cmudict-0.7b.symbols")

    with open(dict_file, "w") as txt:
        txt.write("\n".join(mocked_dictionary))

    with open(symbol_file, "w") as txt:
        txt.write("\n".join(mocked_symbols))

    mocked_data = []
    for ent in mocked_dictionary:
        if len(ent) and ent[:3] == ";;;":
            continue
        word, phones = ent.split("  ")
        if word[0] in string.punctuation:
            # for punctuation, we only keep the puncuation itself
            word = word[0]
        mocked_data.append((word, phones.split(" ")))

    return mocked_data


class TestCMUDict(TempDirMixin, TorchaudioTestCase):
    root_dir = None
    samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = cls.get_base_temp_dir()
        cls.samples = get_mock_dataset(cls.root_dir)

    def _test_cmudict(self, dataset):
        """Test if the dataset is reading the mocked data correctly."""
        n_item = 0
        for i, (word, phones) in enumerate(dataset):
            if dataset.exclude_punctuations == True:
                assert word not in string.punctuation
                # because two punctuations in the beginning are excluded
                i = i + 2
            expected_word, expected_phones = self.samples[i]
            assert word == expected_word
            assert phones == expected_phones
            n_item += 1
        if dataset.exclude_punctuations:
            assert n_item == (len(self.samples) - 2)
        else:
            assert n_item == len(self.samples)

    def test_cmuarctic_path_with_punctuation(self):
        dataset = CMUDict(Path(self.root_dir), exclude_punctuations=False)
        self._test_cmudict(dataset)

    def test_cmuarctic_str_with_punctuation(self):
        dataset = CMUDict(self.root_dir, exclude_punctuations=False)
        self._test_cmudict(dataset)

    def test_cmuarctic_str(self):
        dataset = CMUDict(self.root_dir)
        self._test_cmudict(dataset)

    def test_cmuarctic_path(self):
        dataset = CMUDict(Path(self.root_dir))
        self._test_cmudict(dataset)
