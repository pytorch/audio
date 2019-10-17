import os

import torch.utils.data as data

import torchaudio
from torchaudio.datasets.utils import download, extract, shuffle, walk


def load_librispeech_item(fileid, path, ext_audio, ext_txt):

    speaker, chapter, utterance = fileid.split("-")

    file_text = speaker + "-" + chapter + ext_txt
    file_text = os.path.join(path, speaker, chapter, file_text)
    file_audio = speaker + "-" + chapter + "-" + utterance + ext_audio
    file_audio = os.path.join(path, speaker, chapter, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    for line in open(file_text):
        fileid_text, content = line.strip().split(" ", 1)
        if file_audio == fileid_text:
            break
    else:
        # Translation not found
        raise ValueError

    return {
        "speaker": speaker,
        "chapter": chapter,
        "utterance": utterance,
        "content": content,
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def load_librispeech(fileids):
    """
    Load data corresponding to each LIBRISPEECH fileids.

    Input: path, file name identifying a row of data
    Output: dictionary with id, waveform, sample_rate, translation
    """

    text_extension = ".trans.txt"
    audio_extension = ".flac"

    for data_path, fileid in fileids:
        fileid = os.path.basename(fileid).split(".")[0]
        yield load_librispeech_item(fileid, data_path, audio_extension, text_extension)


def LIBRISPEECH(root, selection):
    """
    Create a generator for LibriSpeech.
    """

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
    # path = shuffle(path)
    # path, l = generator_length(path)
    return load_librispeech(path)


class LIBRISPEECH2(data.Dataset):

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self, root, selection):

        selections = [
            "dev-clean",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]

        assert selection in selections

        ext_archive = ".tar.gz"
        base = "http://www.openslr.org/resources/12/"
        url = os.path.join(base, selection + ext_archive)
        folder_in_archive = os.path.join("LibriSpeech", selection)

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                torchaudio.datasets.utils.download_url(_url, root)
            torchaudio.datasets.utils.extract_archive(archive)

        walker = torchaudio.datasets.utils.walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._list = list(walker)

    def __getitem__(self, n):
        fileid = self._list[n]
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __iter__(self):
        for fileid in self._list:
            yield load_librispeech_item(
                fileid, self._path, self._ext_audio, self._ext_txt
            )

    def __len__(self):
        return len(self._list)
