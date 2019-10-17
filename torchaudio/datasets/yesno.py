import os

import torch.utils.data as data
import torchaudio
from torchaudio.datasets.utils import download, extract, shuffle, walk


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
    Create a generator for YESNO.
    """

    url = [("http://www.openslr.org/resources/1/waves_yesno.tar.gz", "waves_yesno")]

    path = download(url, root_path=root)
    path = extract(path)
    path = walk(path, extension=".wav")
    # path = shuffle(path)
    # path, l = generator_length(path)
    return load_yesno(path)


class YESNO2(data.Dataset):

    _url = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"
    _folder_in_archive = "waves_yesno"

    _ext_audio = ".wav"

    def __init__(self, root):

        # torchaudio.datasets.utils.download_url(self._url, root)

        filename = os.path.basename(self._url)
        filename = os.path.join(root, filename)
        # torchaudio.datasets.utils.extract_archive(filename)

        self._path = os.path.join(root, self._folder_in_archive)

        self._list = torchaudio.datasets.utils.list_files_recursively(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )

    def __getitem__(self, n):

        fileid = self._list[n]

        # Read label
        label = fileid.split("_")

        # Read wav
        file_audio = os.path.join(self._path, fileid + self._ext_audio)
        waveform, sample_rate = torchaudio.load(file_audio)

        return {"label": label, "waveform": waveform, "sample_rate": sample_rate}

    def __len__(self):
        return len(self._list)
