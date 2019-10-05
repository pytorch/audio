import os
from warnings import warn

import torchaudio
from torchaudio.dataset.utils import download, extract, shuffle, walk


def load_vctk(fileids):
    """
    Load data corresponding to each VCTK fileids.

    Input: path, file name identifying a row of data
    Output: dictionary with id, content, waveform, sample_rate
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
    return load_vctk(path)
