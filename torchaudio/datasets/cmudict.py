import os
import re
import string
from pathlib import Path
from typing import Tuple, Union, List

from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url

_CHECKSUMS = {
    "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b":
    "825f4ebd9183f2417df9f067a9cabe86",
    "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols":
    "385e490aabc71b48e772118e3d02923e",
}


class CMUDict(Dataset):
    """Create a Dataset for CMU Pronouncing Dictionary (CMUDict).

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional):
            The URL to download the dictionary from.
            (default: ``"http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"``)
        url_symbols (str, optional):
            The URL to download the list of symbols from.
            (default: ``"http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    def __init__(self,
                 root: Union[str, Path],
                 url: str = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b",
                 url_symbols: str = "http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b.symbols",
                 download: bool = False,
                 exclude_punctuations: bool = True) -> None:

        self.exclude_punctuations = exclude_punctuations

        root = Path(root)
        if not os.path.isdir(root):
            os.mkdir(root)

        if download:
            if os.path.isdir(root):
                checksum = _CHECKSUMS.get(url, None)
                download_url(url, root, hash_value=checksum, hash_type="md5")
                checksum = _CHECKSUMS.get(url_symbols, None)
                download_url(url_symbols, root, hash_value=checksum, hash_type="md5")
            else:
                RuntimeError(f"The argument `root` must be a path to directory, "
                             "but '{root}' is passed in instead.")

        self._root_path = root
        basename = os.path.basename(url)
        basename_symbols = os.path.basename(url_symbols)

        with open(os.path.join(self._root_path, basename_symbols), "r") as text:
            self._symbols = text.readlines()

        with open(os.path.join(self._root_path, basename), "r") as text:
            self._dictionary = self._parse_dictionary(text.readlines())

    def _parse_dictionary(self, lines: List[str]):
        _alt_re = re.compile(r'\([0-9]+\)')
        cmudict: List[Tuple[str, List[str]]] = list()
        for line in lines:
            if not line or line.startswith(';;;'):  # ignore comments
                continue

            word, phones = line.strip().split('  ')
            if word[0] in string.punctuation:
                if self.exclude_punctuations:
                    continue
                else:
                    # !EXCLAMATION-POINT -> !
                    word = word[0]

            # if a word have multiple pronunciations, there will be (number) appended to it
            # for example, DATAPOINTS and DATAPOINTS(1),
            # the regular expression `_alt_re` removes the '(1)' and change the word DATAPOINTS(1) to DATAPOINTS
            word = re.sub(_alt_re, '', word)
            phones = phones.split(" ")
            cmudict.append((word, phones))
        return cmudict

    def __getitem__(self, n: int) -> Tuple[str, List[str]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            tuple: The corresponding word and phonemes ``(word, [phonemes])``.

        """
        return self._dictionary[n]

    def __len__(self) -> int:
        return len(self._dictionary)

    def get_symbols(self) -> List[str]:
        return self._symbols.copy()
