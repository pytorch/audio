import os
import warnings

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive, walk_files

URL = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"
FOLDER_IN_ARCHIVE = "waves_yesno"


def load_yesno_item(fileid, path, ext_audio):
    # Read label
    label = fileid.split("_")

    # Read wav
    file_audio = os.path.join(path, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return {"label": label, "waveform": waveform, "sample_rate": sample_rate}


class YESNO(Dataset):

    _ext_audio = ".wav"

    def __init__(
        self,
        root,
        url=URL,
        folder_in_archive=FOLDER_IN_ARCHIVE,
        download=False,
        transform=None,
        target_transform=None,
        return_dict=False,
    ):

        if not return_dict:
            warnings.warn(
                "In the next version, the item returned will be a dictionary. "
                "Please use `return_dict=True` to enable this behavior now, "
                "and suppress this warning.",
                DeprecationWarning,
            )

        self.transform = transform
        self.target_transform = target_transform
        self.return_dict = return_dict

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        fileid = self._walker[n]
        item = load_yesno_item(fileid, self._path, self._ext_audio)

        waveform = item["waveform"]
        if self.transform is not None:
            waveform = self.transform(waveform)
        item["waveform"] = waveform

        label = item["label"]
        if self.target_transform is not None:
            label = self.target_transform(label)
        item["label"] = label

        if self.return_dict:
            return item

        return item["waveform"], item["label"]

    def __len__(self):
        return len(self._walker)
