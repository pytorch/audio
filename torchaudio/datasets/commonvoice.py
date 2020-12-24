import os
import csv
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset


def load_commonvoice_item(line: List[str],
                          header: List[str],
                          path: str,
                          folder_audio: str,
                          ext_audio: str) -> Tuple[Tensor, int, Dict[str, str]]:
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent

    assert header[1] == "path"
    fileid = line[1]

    filename = os.path.join(path, folder_audio, fileid + ext_audio)

    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(zip(header, line))

    return waveform, sample_rate, dic


class COMMONVOICE(Dataset):
    """Create a Dataset for CommonVoice.

    Args:
        root (str or Path): Path to the directory where the dataset is located.
             (Where the ``tsv`` file is present.)
        tsv (str, optional):
            The name of the tsv file used to construct the metadata, such as
            ``"train.tsv"``, ``"test.tsv"``, ``"dev.tsv"``, ``"invalidated.tsv"``,
            ``"validated.tsv"`` and ``"other.tsv"``. (default: ``"train.tsv"``)
        url (str, optional): Deprecated, not used.
        folder_in_archive (str, optional): Deprecated, not used.
        version (str): Deprecated, not used.
        download (bool, optional): Deprecated, not used.
    """

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self,
                 root: Union[str, Path],
                 tsv: str = "train.tsv",
                 url: Optional[str] = None,
                 folder_in_archive: Optional[str] = None,
                 version: Optional[str] = None,
                 download: Optional[bool] = None) -> None:
        if download:
            raise RuntimeError(
                "Common Voice dataset requires user agreement on the usage term, "
                "and torchaudio no longer provides the download feature. "
                "Please download the dataset and extract it manually.")

        deprecated = [
            ('url', url),
            ('folder_in_archive', folder_in_archive),
            ('version', version),
            ('download', download)
        ]
        for name, val in deprecated:
            if val is not None:
                warnings.warn(
                    f"`{name}` argument is no longer used and deprecated. "
                    "It will be removed in 0.9.0 releaase. "
                    "Please remove it from the function call")

        # Get string representation of 'root' in case Path object is passed
        self._path = os.fspath(root)
        self._tsv = os.path.join(self._path, tsv)

        with open(self._tsv, "r") as tsv_:
            walker = csv.reader(tsv_, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, Dict[str, str]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, dictionary)``,  where dictionary is built
            from the TSV file with the following keys: ``client_id``, ``path``, ``sentence``,
            ``up_votes``, ``down_votes``, ``age``, ``gender`` and ``accent``.
        """
        line = self._walker[n]
        return load_commonvoice_item(line, self._header, self._path, self._folder_audio, self._ext_audio)

    def __len__(self) -> int:
        return len(self._walker)
