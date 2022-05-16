import os
from pathlib import Path

from torchaudio.datasets import CMUDict
from torchaudio_unittest.common_utils import TempDirMixin, TorchaudioTestCase


def get_mock_dataset(root_dir, return_punc=False):
    """
    root_dir: directory to the mocked dataset
    """
    header = [
        ";;; # CMUdict  --  Major Version: 0.07",
        ";;; ",
        ";;; # $HeadURL$",
    ]

    puncs = [
        "!EXCLAMATION-POINT  EH2 K S K L AH0 M EY1 SH AH0 N P OY2 N T",
        '"CLOSE-QUOTE  K L OW1 Z K W OW1 T',
        "#HASH-MARK  HH AE1 M AA2 R K",
        "%PERCENT  P ER0 S EH1 N T",
        "&AMPERSAND  AE1 M P ER0 S AE2 N D",
        "'END-INNER-QUOTE  EH1 N D IH1 N ER0 K W OW1 T",
        "(BEGIN-PARENS  B IH0 G IH1 N P ER0 EH1 N Z",
        ")CLOSE-PAREN  K L OW1 Z P ER0 EH1 N",
        "+PLUS  P L UH1 S",
        ",COMMA  K AA1 M AH0",
        "--DASH  D AE1 SH",
        "!EXCLAMATION-POINT  EH2 K S K L AH0 M EY1 SH AH0 N P OY2 N T",
        "/SLASH  S L AE1 SH",
        ":COLON  K OW1 L AH0 N",
        ";SEMI-COLON  S EH1 M IY0 K OW1 L AH0 N",
        "?QUESTION-MARK  K W EH1 S CH AH0 N M AA1 R K",
        "{BRACE  B R EY1 S",
        "}CLOSE-BRACE  K L OW1 Z B R EY1 S",
        "...ELLIPSIS  IH2 L IH1 P S IH0 S",
    ]

    punc_outputs = [
        "!",
        '"',
        "#",
        "%",
        "&",
        "'",
        "(",
        ")",
        "+",
        ",",
        "--",
        "!",
        "/",
        ":",
        ";",
        "?",
        "{",
        "}",
        "...",
    ]

    words = [
        "3-D  TH R IY1 D IY2",
        "'BOUT  B AW1 T",
        "'CAUSE  K AH0 Z",
        "'TWAS  T W AH1 Z",
        "A  AH0",
        "B  B IY1",
        "C  S IY1",
        "D  D IY1",
        "E  IY1",
        "F  EH1 F",
        "G  JH IY1",
        "H  EY1 CH",
        "I  AY1",
        "J  JH EY1",
        "K  K EY1",
        "L  EH1 L",
        "M  EH1 M",
        "N  EH1 N",
        "O  OW1",
        "P  P IY1",
        "Q  K Y UW1",
        "R  AA1 R",
        "S  EH1 S",
        "T  T IY1",
        "U  Y UW1",
        "V  V IY1",
        "X  EH1 K S",
        "Y  W AY1",
        "Z  Z IY1",
    ]

    mocked_symbols = [
        "AA1",
        "AA2",
        "AE1",
        "AE2",
        "AH0",
        "AH1",
        "AY1",
        "B",
        "CH",
        "D",
        "EH1",
        "EH2",
        "ER0",
        "EY1",
        "F",
        "G",
        "HH",
        "IH0",
        "IH1",
        "IY0",
        "IY1",
        "IY2",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "OW1",
        "OY2",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH1",
        "UW0",
        "UW1",
        "V",
        "W",
        "Y",
        "Z",
    ]

    dict_file = os.path.join(root_dir, "cmudict-0.7b")
    symbol_file = os.path.join(root_dir, "cmudict-0.7b.symbols")

    with open(dict_file, "w") as fileobj:
        for section in [header, puncs, words]:
            for line in section:
                fileobj.write(line)
                fileobj.write("\n")

    with open(symbol_file, "w") as txt:
        txt.write("\n".join(mocked_symbols))

    mocked_data = []

    if return_punc:
        for i, ent in enumerate(puncs):
            _, phones = ent.split("  ")
            mocked_data.append((punc_outputs[i], phones.split(" ")))

    for ent in words:
        word, phones = ent.split("  ")
        mocked_data.append((word, phones.split(" ")))

    return mocked_data


class TestCMUDict(TempDirMixin, TorchaudioTestCase):
    root_dir = None
    root_punc_dir = None
    samples = []
    punc_samples = []

    @classmethod
    def setUpClass(cls):
        cls.root_dir = os.path.join(cls.get_base_temp_dir(), "normal")
        os.mkdir(cls.root_dir)
        cls.samples = get_mock_dataset(cls.root_dir)
        cls.root_punc_dir = os.path.join(cls.get_base_temp_dir(), "punc")
        os.mkdir(cls.root_punc_dir)
        cls.punc_samples = get_mock_dataset(cls.root_punc_dir, return_punc=True)

    def _test_cmudict(self, dataset):
        """Test if the dataset is reading the mocked data correctly."""
        n_item = 0
        for i, (word, phones) in enumerate(dataset):
            expected_word, expected_phones = self.samples[i]
            assert word == expected_word
            assert phones == expected_phones
            n_item += 1
        assert n_item == len(self.samples)

    def _test_punc_cmudict(self, dataset):
        """Test if the dataset is reading the mocked data with punctuations correctly."""
        n_item = 0
        for i, (word, phones) in enumerate(dataset):
            expected_word, expected_phones = self.punc_samples[i]
            assert word == expected_word
            assert phones == expected_phones
            n_item += 1
        assert n_item == len(self.punc_samples)

    def test_cmuarctic_path_with_punctuation(self):
        dataset = CMUDict(Path(self.root_punc_dir), exclude_punctuations=False)
        self._test_punc_cmudict(dataset)

    def test_cmuarctic_str_with_punctuation(self):
        dataset = CMUDict(self.root_punc_dir, exclude_punctuations=False)
        self._test_punc_cmudict(dataset)

    def test_cmuarctic_path(self):
        dataset = CMUDict(Path(self.root_punc_dir), exclude_punctuations=True)
        self._test_cmudict(dataset)

    def test_cmuarctic_str(self):
        dataset = CMUDict(self.root_punc_dir, exclude_punctuations=True)
        self._test_cmudict(dataset)
