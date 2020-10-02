import os
from typing import List, Dict, Tuple

import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader
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
_CHECKSUMS = {
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/tt.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/en.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/de.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/fr.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/cy.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/br.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/cv.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/tr.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ky.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ga-IE.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/kab.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ca.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/zh-TW.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/sl.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/it.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/nl.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/cnh.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/eo.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/et.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/fa.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/eu.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/es.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/zh-CN.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/mn.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/sah.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/dv.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/rw.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/sv-SE.tar.gz":
    None,
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/ru.tar.gz":
    None
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
        root (str): Path to the directory where the dataset is found or downloaded.
        tsv (str, optional): The name of the tsv file used to construct the metadata.
            (default: ``"train.tsv"``)
        url (str, optional): The language of the dataset to download.
            This is **NOT** the actual URL. (default: ``"english"``).
            Valid choices are ``"tatar"``, ``"english"``, ``"german"``, ``"french"``,
            ``"welsh"``, ``"breton"``, ``"chuvash"``, ``"turkish"``, ``"kyrgyz"``,
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
                 root: str,
                 tsv: str = TSV,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 version: str = VERSION,
                 download: bool = False) -> None:

        languages = {
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

        if url in languages:
            ext_archive = ".tar.gz"
            language = languages[url]

            base_url = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com"
            url = os.path.join(base_url, version, language + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, version, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        self._tsv = os.path.join(root, folder_in_archive, tsv)

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
