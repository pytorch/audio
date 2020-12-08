import os
import warnings
from typing import List, Dict, Tuple, Optional

import torchaudio
from torchaudio.datasets.utils import extract_archive, unicode_csv_reader, validate_file
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
LANGUAGE = "english"
VERSION = "cv-corpus-5.1-2020-06-22"
TSV = "train.tsv"
_CHECKSUMS = {
    "cv-corpus-5.1-2020-06-22/tt.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/en.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/de.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/fr.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/cy.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/br.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/cv.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/tr.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/ky.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/ga-IE.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/kab.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/ca.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/zh-TW.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/sl.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/it.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/nl.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/cnh.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/eo.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/et.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/fa.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/eu.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/es.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/zh-CN.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/mn.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/sah.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/dv.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/rw.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/sv-SE.tar.gz": None,
    "cv-corpus-5.1-2020-06-22/ru.tar.gz": None,
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
    """Create a Dataset for `CommonVoice <https://commonvoice.mozilla.org/>`_.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        tsv (str, optional): The name of the tsv file used to construct the metadata.
            (default: ``"train.tsv"``)
        url (str, optional): Deprecated.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        version (str): Version string. (default: ``"cv-corpus-5.1-2020-06-22"``)
        download (bool, optional): Deprecated.
        language (str, optional): Language of the dataset. (default: None)
            The following values are mapped to their corresponding shortened version:
            ``"tatar"``, ``"english"``, ``"german"``,
            ``"french"``, ``"welsh"``, ``"breton"``, ``"chuvash"``, ``"turkish"``, ``"kyrgyz"``,
            ``"irish"``, ``"kabyle"``, ``"catalan"``, ``"taiwanese"``, ``"slovenian"``,
            ``"italian"``, ``"dutch"``, ``"hakha chin"``, ``"esperanto"``, ``"estonian"``,
            ``"persian"``, ``"portuguese"``, ``"basque"``, ``"spanish"``, ``"chinese"``,
            ``"mongolian"``, ``"sakha"``, ``"dhivehi"``, ``"kinyarwanda"``, ``"swedish"``,
            ``"russian"``, ``"indonesian"``, ``"arabic"``, ``"tamil"``, ``"interlingua"``,
            ``"latvian"``, ``"japanese"``, ``"votic"``, ``"abkhaz"``, ``"cantonese"`` and
            ``"romansh sursilvan"``.
            For the other allowed values, Please checkout https://commonvoice.mozilla.org/en/datasets.
    """

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self,
                 root: str,
                 tsv: str = TSV,
                 url: Optional[str] = None,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 version: str = VERSION,
                 download: Optional[bool] = False,
                 language: str = LANGUAGE) -> None:

        if download is True:
            raise RuntimeError(
                "The dataset is no longer publicly accessible. You need to "
                "download the archives externally and place them in the root "
                "directory."
            )
        elif download is False:
            warnings.warn(
                "The use of the download flag is deprecated, since the dataset "
                "is no longer directly accessible.", RuntimeWarning
            )

        if url is not None:
            warnings.warn(
                "The use of the url flag is deprecated, since the dataset "
                "is no longer publicly accessible. To specify the language of the dataset, "
                "please use the language parameter instead.", RuntimeWarning
            )

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

        if language in languages:
            ext_archive = ".tar.gz"
            language = languages[language]
            url = os.path.join(version, language + ext_archive)
        else:
            raise ValueError(
                'Allowed language values are "tatar", "english", "german",'
                '"french", "welsh", "breton", "chuvash", "turkish", "kyrgyz",'
                '"irish", "kabyle", "catalan", "taiwanese", "slovenian",'
                '"italian", "dutch", "hakha chin", "esperanto", "estonian",'
                '"persian", "portuguese", "basque", "spanish", "chinese",'
                '"mongolian", "sakha", "dhivehi", "kinyarwanda", "swedish",'
                '"russian", "indonesian", "arabic", "tamil", "interlingua",'
                '"latvian", "japanese", "votic", "abkhaz", "cantonese" and'
                '"romansh sursilvan".'
            )

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, version, basename)

        self._path = os.path.join(root, folder_in_archive)

        if not os.path.isdir(self._path):
            if os.path.isfile(archive):
                checksum = _CHECKSUMS.get(url, None)
                if checksum:
                    filepath = os.path.basename(url)
                    with open(filepath, "rb") as file_obj:
                        if not validate_file(file_obj, checksum, "sha256"):
                            raise RuntimeError(
                                f"The hash of {filepath} does not match. Delete the file manually and retry."
                            )
                extract_archive(archive)
            else:
                raise RuntimeError(
                    "The dataset is no longer publicly accessible. You need to "
                    "download the archives externally and place them in the root "
                    "directory."
                )

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
