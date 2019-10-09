import os

import torchaudio
from torchaudio.datasets.utils import download, extract, shuffle, walk


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
            # Translation not found
            raise ValueError

        yield {
            "id": fileid,
            "content": content,
            "waveform": waveform,
            "sample_rate": sample_rate,
        }


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
