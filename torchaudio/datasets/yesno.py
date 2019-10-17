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


class YESNO(data.IterableDataset):

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
