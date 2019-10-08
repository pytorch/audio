import os

import torchaudio
from torchaudio.dataset.utils import download, extract, shuffle, walk


def load_commonvoice(fileids, tsv):
    """
    Load data corresponding to each COMMONVOICE fileids.

    Input: path, file name identifying a row of data
    Output: dictionary with keys from tsv along with, waveform, sample_rate
    """

    for path, fileid in fileids:
        filename = os.path.join(path, "clips", fileid)
        tsv = os.path.join(path, tsv)

        found = False
        with open(tsv) as tsv:
            header = next(tsv).strip().split("\t")
            for line in tsv:
                if fileid in line:
                    # client_id, path, sentence, up_votes, down_votes, age, gender, accent
                    line = line.strip().split("\t")
                    found = True
                    break
            if not found:
                continue

        waveform, sample_rate = torchaudio.load(filename)

        dic = dict(zip(header, line))
        dic["waveform"] = waveform
        dic["sample_rate"] = sample_rate

        yield dic


def COMMONVOICE(root, language="tatar", tsv="train.tsv"):
    """
    Cache a pipeline loading COMMONVOICE.
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
    path = shuffle(path)
    return load_commonvoice(path, tsv=tsv)
