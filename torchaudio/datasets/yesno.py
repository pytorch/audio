import os

import torchaudio
from torchaudio.dataset.utils import download, extract, shuffle, walk


def load_yesno(fileids):
    """
    Load data corresponding to each YESNO fileids.

    Input: path, file name identifying a row of data
    Output: dictinoary with label, waveform, sample_rate
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
    return load_yesno(path)
