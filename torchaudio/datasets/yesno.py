import os

import torchaudio
from torchaudio.datasets.utils import data, download_url, extract_archive, walk_files

DEFAULT_URL = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"


def load_yesno_item(fileid, path, ext_audio):
    # Read label
    label = fileid.split("_")

    # Read wav
    file_audio = os.path.join(path, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return {"label": label, "waveform": waveform, "sample_rate": sample_rate}


class YESNO(data.Dataset):

    _ext_audio = ".wav"

    def __init__(self, root, url=DEFAULT_URL):

        folder_in_archive = "waves_yesno"

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                download_url(url, root)
            extract_archive(archive)

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        fileid = self._walker[n]
        return load_yesno_item(fileid, self._path, self._ext_audio)

    def __len__(self):
        return len(self._walker)
