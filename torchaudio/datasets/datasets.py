import os
import pickle
import random
from functools import partial, reduce
from warnings import warn

import torch

import torchaudio


class Cache:
    """
    Wrap a generator so that, whenever a new item is returned, it is saved to disk in a pickle.
    """

    def __init__(self, generator, location):
        self.generator = generator
        self.location = location

        self._id = id(self)
        self._cache = []
        self._internal_index = 0

    def __iter__(self):
        self._internal_index = 0
        return self

    def __next__(self):
        if self._internal_index < len(self):
            item = self[self._internal_index]
        else:
            item = next(self.generator)

            file = str(self._id) + "-" + str(len(self))
            file = os.path.join(self.location, file)
            self._cache.append(file)

            os.makedirs(self.location, exist_ok=True)
            with open(file, "wb") as file:
                pickle.dump(item, file)

        self._internal_index += 1
        return item

    def __getitem__(self, index):
        file = self._cache[index]
        with open(file, "rb") as file:
            item = pickle.load(file)
        return item

    def __len__(self):
        # Return length of cache
        return len(self._cache)


class Buffer:
    """
    Wrap a generator so as to keep the last few in memory.
    """

    def __init__(self, generator, capacity=10):
        self.generator = generator
        self.capacity = capacity
        self._cache = []
        self._fill()

    def _fill(self):
        while len(self._cache) <= self.capacity:
            self._cache.append(next(self.generator))

    def __getitem__(self, n):
        self._fill()
        return self._cache[n]

    def __iter__(self):
        return self

    def __next__(self):
        item = self._cache.pop(0)
        self._fill()
        return item


def download(urls, root_path):
    """
    Download each url to root_path.

    Input: url generator, folder inside archive
    Output: downloaded archive, folder inside archive
    """
    for url, folder in urls:
        torchaudio.datasets.utils.download_url(url, root_path)
        file = os.path.join(root_path, os.path.basename(url))
        yield file, folder


def extract(files):
    """
    Extract each archive to their respective folder.

    Input: (url, folder name inside archive) generator
    Output: path to inside archive
    """
    for file, folder in files:
        torchaudio.datasets.utils.extract_archive(file)
        path = os.path.dirname(file)
        path = os.path.join(path, folder)
        yield path


def walk(paths, extension):
    """
    Walk inside a path recursively to find all files with given extension.

    Input: path
    Output: path, file name identifying a row of data
    """
    for path in paths:
        for dp, dn, fn in os.walk(path):
            for f in fn:
                if extension in f:
                    yield path, f


def shuffle(generator):
    """
    Shuffle the order of a generator.

    Input: generator
    Output: generator
    """

    # Load whole generator in memory
    generator = list(generator)
    # print(len(generator))
    random.shuffle(generator)
    for g in generator:
        yield g


def filtering(fileids, reference):
    """
    Skip fileids that are not present in given reference file.

    Output: (path, file) generator, reference file
    Output: path, file
    """

    path_old = ""

    for path, fileid in fileids:

        # Check if same path to avoid reloading the file constantly
        if path != path_old:
            ref = os.path.join(path, reference)
            with open(ref) as ref:
                r = "".join(ref.readlines())
            path_old = path

        # It would be more efficient to loop through the reference file instead
        if fileid in r:
            yield path, fileid


def load_yesno(fileids):
    """
    Load data corresponding to each YESNO fileids.

    Input: path, file name identifying a row of data
    Output: label, waveform, sample_rate
    """

    extension = ".wav"
    for path, fileid in fileids:
        file = os.path.join(path, fileid)
        waveform, sample_rate = torchaudio.load(file)
        label = os.path.basename(fileid).split(".")[0].split("_")

        yield {"label": label, "waveform": waveform, "sample_rate": sample_rate}


def YESNO(root):
    """
    Cache a pipeline loading YESNO.
    """

    url = [("http://www.openslr.org/resources/1/waves_yesno.tar.gz", "waves_yesno")]

    path = download(url, root_path=root)
    path = extract(path)
    path = walk(path, extension=".wav")
    path = shuffle(path)
    data = load_yesno(path)

    # return Buffer(data)
    return Cache(data, "tmp/")


def load_vctk(fileids):
    """
    Load data corresponding to each VCTK fileids.

    Input: path, file name identifying a row of data
    Output: id, content, waveform, sample_rate
    """

    txt_folder = "txt"
    txt_extension = ".txt"

    audio_folder = "wav48"
    audio_extension = ".wav"

    for path, fileid in fileids:

        fileid = os.path.basename(fileid).split(".")[0]
        folder = fileid.split("_")[0]
        txt_file = os.path.join(path, txt_folder, folder, fileid + txt_extension)
        audio_file = os.path.join(path, audio_folder, folder, fileid + audio_extension)

        try:
            with open(txt_file) as txt_file:
                content = txt_file.readlines()[0]
        except FileNotFoundError:
            warn("Translation not found for {}".format(audio_file))
            # warn("File not found: {}".format(txt_file))
            continue

        waveform, sample_rate = torchaudio.load(audio_file)

        yield {
            "id": fileid,
            "content": content,
            "waveform": waveform,
            "sample_rate": sample_rate,
        }


def VCTK(root):
    """
    Cache a pipeline loading VCTK.
    """

    url = [
        (
            "http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz",
            "VCTK-Corpus/",
        )
    ]

    path = download(url, root_path=root)
    path = extract(path)
    path = walk(path, extension=".wav")
    path = shuffle(path)
    data = load_vctk(path)

    return Cache(data, "tmp/")


def load_librispeech(fileids):
    """
    Load data corresponding to each LIBRISPEECH fileids.

    Input: path, file name identifying a row of data
    Output: id, waveform, sample_rate, translation
    """

    text_extension = ".trans.txt"
    audio_extension = ".flac"
    for data_path, fileid in fileids:
        fileid = os.path.basename(fileid).split(".")[0]
        folder1, folder2, file = fileid.split("-")
        file_text = folder1 + "-" + folder2 + text_extension
        file_text = os.path.join(data_path, folder1, folder2, file_text)
        file_audio = folder1 + "-" + folder2 + "-" + file + audio_extension
        file_audio = os.path.join(data_path, folder1, folder2, file_audio)
        waveform, sample_rate = torchaudio.load(file_audio)

        found = False
        for line in open(file_text):
            fileid_text, content = line.strip().split(" ", 1)
            if fileid == fileid_text:
                found = True
                break
        if not found:
            from warnings import warn

            warn("Translation not found for {}.".format(fileid))
            continue

        yield {
            "id": fileid,
            "content": content,
            "waveform": waveform,
            "sample_rate": sample_rate,
        }


def LIBRISPEECH(root, selection="dev-clean"):
    """
    Cache a pipeline loading LIBRISPEECH.
    """

    # http://www.openslr.org/resources/12/dev-clean.tar.gz
    # http://www.openslr.org/resources/12/test-clean.tar.gz
    # http://www.openslr.org/resources/12/test-other.tar.gz
    # http://www.openslr.org/resources/12/train-clean-100.tar.gz
    # http://www.openslr.org/resources/12/train-clean-360.tar.gz
    # http://www.openslr.org/resources/12/train-other-500.tar.gz

    selections = [
        "dev-clean",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]

    base = "http://www.openslr.org/resources/12/"
    url = [
        (
            os.path.join(base, selection + ".tar.gz"),
            os.path.join("LibriSpeech", selection),
        )
    ]

    path = download(url, root_path=root)
    path = extract(path)
    path = walk(path, extension=".flac")
    path = shuffle(path)
    data = load_librispeech(path)

    return Cache(data, "tmp/")


def load_commonvoice(fileids, tsv):
    """
    Load data corresponding to each COMMONVOICE fileids.

    Input: path, file name identifying a row of data
    Output: client_id, path, sentence, up_votes, down_votes, age, gender, accent, waveform, sample_rate
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
    path = filtering(path, reference=tsv)
    data = load_commonvoice(path, tsv=tsv)

    return Cache(data, "tmp/")
