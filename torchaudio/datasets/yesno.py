import os

import torch.utils.data as data

import torchaudio
from torchaudio.datasets.utils import download, extract, shuffle, walk


def load_yesno_item(fileid, path, ext_audio):
    # Read label
    label = fileid.split("_")

    # Read wav
    file_audio = os.path.join(path, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return {"label": label, "waveform": waveform, "sample_rate": sample_rate}


def load_yesno(fileids):
    """
    Load data corresponding to each YESNO fileids.

    Input: path, file name identifying a row of data
    Output: dictinoary with label, waveform, sample_rate
    """

    extension = ".wav"
    for path, fileid in fileids:
        fileid = os.path.basename(fileid).split(".")[0]
        yield load_yesno_item(fileid, path, extension)


def YESNO(root):
    """
    Create a generator for YESNO.
    """

    url = [("http://www.openslr.org/resources/1/waves_yesno.tar.gz", "waves_yesno")]

    path = download(url, root_path=root)
    path = extract(path)
    path = walk(path, extension=".wav")
    # path = shuffle(path)
    # path, l = generator_length(path)
    return load_yesno(path)


class YESNO2(data.IterableDataset):

    _ext_audio = ".wav"

    def __init__(self, root):

        url = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"
        folder_in_archive = "waves_yesno"

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
        self._walker = walker

    def __iter__(self):
        for fileid in self._walker:
            yield load_yesno_item(fileid, self._path, self._ext_audio)

