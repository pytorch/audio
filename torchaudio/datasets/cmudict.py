import os
import re
from pathlib import Path
from typing import Iterable, Tuple, Union, List

from torch.hub import download_url_to_file
from torch.utils.data import Dataset

_CHECKSUMS = {
    "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b": "209a8b4cd265013e96f4658632a9878103b0c5abf62b50d4ef3ae1be226b29e4",  # noqa: E501
    "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols": "408ccaae803641c6d7b626b6299949320c2dbca96b2220fd3fb17887b023b027",  # noqa: E501
}
_PUNCTUATIONS = set(
    [
        "!EXCLAMATION-POINT",
        '"CLOSE-QUOTE',
        '"DOUBLE-QUOTE',
        '"END-OF-QUOTE',
        '"END-QUOTE',
        '"IN-QUOTES',
        '"QUOTE',
        '"UNQUOTE',
        "#HASH-MARK",
        "#POUND-SIGN",
        "#SHARP-SIGN",
        "%PERCENT",
        "&AMPERSAND",
        "'END-INNER-QUOTE",
        "'END-QUOTE",
        "'INNER-QUOTE",
        "'QUOTE",
        "'SINGLE-QUOTE",
        "(BEGIN-PARENS",
        "(IN-PARENTHESES",
        "(LEFT-PAREN",
        "(OPEN-PARENTHESES",
        "(PAREN",
        "(PARENS",
        "(PARENTHESES",
        ")CLOSE-PAREN",
        ")CLOSE-PARENTHESES",
        ")END-PAREN",
        ")END-PARENS",
        ")END-PARENTHESES",
        ")END-THE-PAREN",
        ")PAREN",
        ")PARENS",
        ")RIGHT-PAREN",
        ")UN-PARENTHESES",
        "+PLUS",
        ",COMMA",
        "--DASH",
        "-DASH",
        "-HYPHEN",
        "...ELLIPSIS",
        ".DECIMAL",
        ".DOT",
        ".FULL-STOP",
        ".PERIOD",
        ".POINT",
        "/SLASH",
        ":COLON",
        ";SEMI-COLON",
        ";SEMI-COLON(1)",
        "?QUESTION-MARK",
        "{BRACE",
        "{LEFT-BRACE",
        "{OPEN-BRACE",
        "}CLOSE-BRACE",
        "}RIGHT-BRACE",
    ]
)


def _parse_dictionary(lines: Iterable[str], exclude_punctuations: bool) -> List[str]:
    _alt_re = re.compile(r"\([0-9]+\)")
    cmudict: List[Tuple[str, List[str]]] = list()
    for line in lines:
        if not line or line.startswith(";;;"):  # ignore comments
            continue

        word, phones = line.strip().split("  ")
        if word in _PUNCTUATIONS:
            if exclude_punctuations:
                continue
            # !EXCLAMATION-POINT -> !
            # --DASH -> --
            # ...ELLIPSIS -> ...
            if word.startswith("..."):
                word = "..."
            elif word.startswith("--"):
                word = "--"
            else:
                word = word[0]

        # if a word have multiple pronunciations, there will be (number) appended to it
        # for example, DATAPOINTS and DATAPOINTS(1),
        # the regular expression `_alt_re` removes the '(1)' and change the word DATAPOINTS(1) to DATAPOINTS
        word = re.sub(_alt_re, "", word)
        phones = phones.split(" ")
        cmudict.append((word, phones))

    return cmudict


class CMUDict(Dataset):
    """Create a Dataset for CMU Pronouncing Dictionary (CMUDict).

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        exclude_punctuations (bool, optional):
            When enabled, exclude the pronounciation of punctuations, such as
            `!EXCLAMATION-POINT` and `#HASH-MARK`.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str, optional):
            The URL to download the dictionary from.
            (default: ``"http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"``)
        url_symbols (str, optional):
            The URL to download the list of symbols from.
            (default: ``"http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols"``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        exclude_punctuations: bool = True,
        *,
        download: bool = False,
        url: str = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b",
        url_symbols: str = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols",
    ) -> None:

        self.exclude_punctuations = exclude_punctuations

        self._root_path = Path(root)
        if not os.path.isdir(self._root_path):
            raise RuntimeError(f"The root directory does not exist; {root}")

        dict_file = self._root_path / os.path.basename(url)
        symbol_file = self._root_path / os.path.basename(url_symbols)
        if not os.path.exists(dict_file):
            if not download:
                raise RuntimeError(
                    "The dictionary file is not found in the following location. "
                    f"Set `download=True` to download it. {dict_file}"
                )
            checksum = _CHECKSUMS.get(url, None)
            download_url_to_file(url, dict_file, checksum)
        if not os.path.exists(symbol_file):
            if not download:
                raise RuntimeError(
                    "The symbol file is not found in the following location. "
                    f"Set `download=True` to download it. {symbol_file}"
                )
            checksum = _CHECKSUMS.get(url_symbols, None)
            download_url_to_file(url_symbols, symbol_file, checksum)

        with open(symbol_file, "r") as text:
            self._symbols = [line.strip() for line in text.readlines()]

        with open(dict_file, "r", encoding="latin-1") as text:
            self._dictionary = _parse_dictionary(text.readlines(), exclude_punctuations=self.exclude_punctuations)

    def __getitem__(self, n: int) -> Tuple[str, List[str]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            (str, List[str]): The corresponding word and phonemes ``(word, [phonemes])``.

        """
        return self._dictionary[n]

    def __len__(self) -> int:
        return len(self._dictionary)

    @property
    def symbols(self) -> List[str]:
        """list[str]: A list of phonemes symbols, such as `AA`, `AE`, `AH`."""
        return self._symbols.copy()
