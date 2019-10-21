import os

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive, unicode_csv_reader


def load_commonvoice_item(line, header, path, folder_audio):
    fileid = line[1]

    filename = os.path.join(path, folder_audio, fileid)

    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(zip(header, line))
    dic["waveform"] = waveform
    dic["sample_rate"] = sample_rate

    return dic


class COMMONVOICE(Dataset):

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self, root, tsv, language=None, url=None):

        if url is None:
            ext_archive = ".tar.gz"
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
            language = languages.get(language, language)

            base_url = (
                "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4"
                + ".s3.amazonaws.com/cv-corpus-3/"
            )
            url = base_url + language + ext_archive

        tsvs = [
            "dev.tsv",
            "invalidated.tsv",
            "other.tsv",
            "test.tsv",
            "train.tsv",
            "validated.tsv",
        ]

        assert tsv in tsvs

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = root

        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                download_url(url, root)
            extract_archive(archive)

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
