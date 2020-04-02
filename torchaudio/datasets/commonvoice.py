import os
from urllib.parse import urlparse

from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader

# Default TSV should be one of
# dev.tsv
# invalidated.tsv
# other.tsv
# test.tsv
# train.tsv
# validated.tsv

FOLDER_IN_ARCHIVE = "CommonVoice"
LANGUAGE = "english"
VERSION = "cv-corpus-4-2019-12-10"
TSV = "train.tsv"


def load_commonvoice_item(line, header, path, folder_audio):
    # Each line as the following data:
    # client_id, path, sentence, up_votes, down_votes, age, gender, accent

    assert header[1] == "path"
    fileid = line[1]

    filename = os.path.join(path, folder_audio, fileid)

    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(zip(header, line))

    return waveform, sample_rate, dic


class COMMONVOICE(Dataset):
    """
    Create a Dataset for CommonVoice. Each item is a tuple of the form:
    (waveform, sample_rate, dictionary)
    where dictionary is a dictionary built from the tsv file with the following keys:
    client_id, path, sentence, up_votes, down_votes, age, gender, accent.
    """

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self, root, tsv=TSV, language=LANGUAGE, folder_in_archive=FOLDER_IN_ARCHIVE, version=VERSION, download=False):

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
        }

        if language in languages:
            ext_archive = ".tar.gz"
            language = languages[language]

            base_url = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com"
            url = os.path.join(base_url, version, language + ext_archive)

        basename = os.path.basename(urlparse(url).path)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, version, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive, self._path)

        self._tsv = os.path.join(root, tsv)

        with open(self._tsv, "r") as tsv:
            walker = unicode_csv_reader(tsv, delimiter="\t")
            self._header = next(walker)
            self._walker = list(walker)

    def __getitem__(self, n):
        line = self._walker[n]
        return load_commonvoice_item(line, self._header, self._path, self._folder_audio)

    def __len__(self):
        return len(self._walker)
