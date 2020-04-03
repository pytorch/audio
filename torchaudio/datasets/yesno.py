import os
import warnings

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive, walk_files, get_checksum_dict

URL = "http://www.openslr.org/resources/1/waves_yesno.tar.gz"
FOLDER_IN_ARCHIVE = "waves_yesno"


def load_yesno_item(fileid, path, ext_audio):
    # Read label
    labels = [int(c) for c in fileid.split("_")]

    # Read wav
    file_audio = os.path.join(path, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, labels


class YESNO(Dataset):
    """
    Create a Dataset for YesNo. Each item is a tuple of the form:
    (waveform, sample_rate, labels)
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root,
        url=URL,
        folder_in_archive=FOLDER_IN_ARCHIVE,
        download=False,
        transform=None,
        target_transform=None,
    ):

        if transform is not None or target_transform is not None:
            warnings.warn(
                "In the next version, transforms will not be part of the dataset. "
                "Please remove the option `transform=True` and "
                "`target_transform=True` to suppress this warning."
            )

        self.transform = transform
        self.target_transform = target_transform

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if download:
            checksum = get_checksum_dict(dataset=__class__.__name__)
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root, hash_value=checksum[url], hash_type="md5")
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

        # TODO Upon deprecation, uncomment line below and remove following code
        # return item

        waveform, sample_rate, labels = item
        if self.transform is not None:
            waveform = self.transform(waveform)
        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return waveform, sample_rate, labels

    def __len__(self):
        return len(self._walker)
