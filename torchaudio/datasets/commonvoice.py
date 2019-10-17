import os

import torch.utils.data as data

import torchaudio
from torchaudio.datasets.utils import (
    download,
    extract,
    shuffle,
    unicode_csv_reader,
    walk,
)


def load_commonvoice(fileids, tsv_file):
    """
    Load data corresponding to each COMMONVOICE fileids.

    Input: path, file name identifying a row of data
    Output: dictionary with keys from tsv along with, waveform, sample_rate
    """

    for path, fileid in fileids:
        filename = os.path.join(path, "clips", fileid)
        tsv = os.path.join(path, tsv_file)

        with open(tsv) as tsv:
            first_line = True
            for line in unicode_csv_reader(tsv, delimiter="\t"):
                if first_line:
                    header = line
                    first_line = False
                    continue
                if fileid in line:
                    break
            else:
                continue

        waveform, sample_rate = torchaudio.load(filename)

        dic = dict(zip(header, line))
        dic["waveform"] = waveform
        dic["sample_rate"] = sample_rate

        yield dic


def COMMONVOICE(root, language, tsv_file):
    """
    Create a generator for CommonVoice.
    """

    web = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/"

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

    url = web + languages[language] + ".tar.gz"
    url = [(url, "")]

    path = download(url, root_path=root)
    path = extract(path)
    path = walk(path, extension=".mp3")
    # path = shuffle(path)
    # path, l = generator_length(path)
    return load_commonvoice(path, tsv_file)


def load_commonvoice_item(line, header, path, folder_audio):
    fileid = line[1]

    filename = os.path.join(path, folder_audio, fileid)

    waveform, sample_rate = torchaudio.load(filename)

    dic = dict(zip(header, line))
    dic["waveform"] = waveform
    dic["sample_rate"] = sample_rate

    return dic


class COMMONVOICE2(data.IterableDataset):

    _ext_txt = ".txt"
    _ext_audio = ".mp3"
    _folder_audio = "clips"

    def __init__(self, root, language, tsv):

        base = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-3/"

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

        ext_archive = ".tar.gz"
        url = base + languages[language] + ext_archive

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = root

        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                torchaudio.datasets.utils.download_url(url, root)
            torchaudio.datasets.utils.extract_archive(archive)

        # Read header and all lines in tsv file
        tsv = os.path.join(root, tsv)
        with open(tsv) as tsv:
            walker = unicode_csv_reader(tsv, delimiter="\t")
            self._header = next(walker)
            self._list = list(walker)

    def __getitem__(self, n):
        line = self._list[n]
        return load_commonvoice_item(line, self._header, self._path, self._folder_audio)

    def __iter__(self):
        for line in self._list:
            yield load_commonvoice_item(
                line, self._header, self._path, self._folder_audio
            )

    def __len__(self):
        return len(self._list)
