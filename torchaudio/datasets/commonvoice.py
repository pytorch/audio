import os
from pathlib import Path
from typing import List, Dict, Tuple, Union

import torchaudio
from torchaudio.datasets.utils import unicode_csv_reader, extract_archive
from torch import Tensor
from torch.utils.data import Dataset

# Default TSV should be one of
# dev.tsv
# invalidated.tsv
# other.tsv
# test.tsv
# train.tsv
# validated.tsv

FOLDER_IN_ARCHIVE = "CommonVoice"
URL = "english"
VERSION = "cv-corpus-4-2019-12-10"
TSV = "train.tsv"

_LANG_CODE = {
    "tatar": "tt",
    "english": "en",
    "german": "de",
    "french": "fr",
    "welsh": "cy",
    "breton": "br",
    "chuvash": "cv",
    "turkish": "tr",
    "kyrgyz": "ky",
    "irish": "ga-IE",
    "kabyle": "kab",
    "catalan": "ca",
    "taiwanese": "zh-TW",
    "slovenian": "sl",
    "italian": "it",
    "dutch": "nl",
    "hakha chin": "cnh",
    "esperanto": "eo",
    "estonian": "et",
    "persian": "fa",
    "portuguese": "pt",
    "basque": "eu",
    "spanish": "es",
    "chinese": "zh-CN",
    "mongolian": "mn",
    "sakha": "sah",
    "dhivehi": "dv",
    "kinyarwanda": "rw",
    "swedish": "sv-SE",
    "russian": "ru",
    "indonesian": "id",
    "arabic": "ar",
    "tamil": "ta",
    "interlingua": "ia",
    "latvian": "lv",
    "japanese": "ja",
    "votic": "vot",
    "abkhaz": "ab",
    "cantonese": "zh-HK",
    "romansh sursilvan": "rm-sursilv"
}


def load_commonvoice_item(line: List[str],
                          header: List[str],
                          path: str,
                          folder_audio: str) -> Tuple[Tensor, int, Dict[str, str]]:
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent

    assert header[1] == "path"
    fileid = line[1]

    filename = os.path.join(path, folder_audio, fileid)

    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(zip(header, line))

    return waveform, sample_rate, dic


class COMMONVOICE(Dataset):
    """Create a Dataset for CommonVoice.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        tsv (str, optional): The name of the tsv file used to construct the metadata.
            (default: ``"train.tsv"``)
        url (str, optional): The URL to download the dataset from, or the language of
            the dataset to download. (default: ``"english"``).
            Allowed language values are ``"tatar"``, ``"english"``, ``"german"``,
            ``"french"``, ``"welsh"``, ``"breton"``, ``"chuvash"``, ``"turkish"``, ``"kyrgyz"``,
            ``"irish"``, ``"kabyle"``, ``"catalan"``, ``"taiwanese"``, ``"slovenian"``,
            ``"italian"``, ``"dutch"``, ``"hakha chin"``, ``"esperanto"``, ``"estonian"``,
            ``"persian"``, ``"portuguese"``, ``"basque"``, ``"spanish"``, ``"chinese"``,
            ``"mongolian"``, ``"sakha"``, ``"dhivehi"``, ``"kinyarwanda"``, ``"swedish"``,
            ``"russian"``, ``"indonesian"``, ``"arabic"``, ``"tamil"``, ``"interlingua"``,
            ``"latvian"``, ``"japanese"``, ``"votic"``, ``"abkhaz"``, ``"cantonese"`` and
            ``"romansh sursilvan"``.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        version (str): Version string. (default: ``"cv-corpus-4-2019-12-10"``)
            For the other allowed values, Please checkout https://commonvoice.mozilla.org/en/datasets.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self,
                 root: Union[str, Path],
                 tsv: str = TSV,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 version: str = VERSION,
                 download: bool = False) -> None:
        if download:
            raise RuntimeError(
                "Common Voice dataset requires user agreement on the usage term, "
                "and torchaudio no longer provides the download feature. "
                "Please download the dataset manually and extract it in the root directory, "
                "then provide the target language to `url` argument.")
        if url not in _LANG_CODE:
            raise ValueError(f"`url` must be one of available languages: {_LANG_CODE.keys()}")

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        lang_code = _LANG_CODE[url]
        folder_in_archive = os.path.join(folder_in_archive, version, lang_code)

        self._path = os.path.join(root, folder_in_archive)
        self._tsv = os.path.join(root, folder_in_archive, tsv)

        if not os.path.isdir(self._path):
            archive = os.path.join(root, f'{lang_code}.tar')
            if not os.path.isfile(archive):
                raise RuntimeError(
                    f'Neither the data set directory "{folder_in_archive}", '
                    f'or the archive "{lang_code}.tar" is found.')
            extract_archive(archive)
            if not os.path.isdir(self._path):
                raise RuntimeError(
                    f'The data set directory "{folder_in_archive}" is not found after '
                    f'extracting "{lang_code}.tar" is found. '
                    'Is the `version` matching to the archive?')

        with open(self._tsv, "r") as tsv:
            walker = unicode_csv_reader(tsv, delimiter="\t")
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
        return load_commonvoice_item(line, self._header, self._path, self._folder_audio)

    def __len__(self) -> int:
        return len(self._walker)
