import os
import warnings
from typing import Any, Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files
)

URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
FOLDER_IN_ARCHIVE = "VCTK-Corpus"
FOLDER_IN_ARCHIVE_092 = "VCTK-Corpus-0.92"
_CHECKSUMS = {
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip":
    "8a6ba2946b36fcbef0212cad601f4bfa"
}


def load_vctk_item(fileid: str,
                   path: str,
                   ext_audio: str,
                   ext_txt: str,
                   folder_audio: str,
                   folder_txt: str,
                   downsample: bool = False) -> Tuple[Tensor, int, str, str, str]:
    speaker_id, utterance_id = fileid.split("_")

    # Read text
    file_txt = os.path.join(path, folder_txt, speaker_id, fileid + ext_txt)
    with open(file_txt) as file_text:
        utterance = file_text.readlines()[0]

    # Read wav
    file_audio = os.path.join(path, folder_audio, speaker_id, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)
    if downsample:
        # TODO Remove this parameter after deprecation
        F = torchaudio.functional
        T = torchaudio.transforms
        # rate
        sample = T.Resample(sample_rate, 16000, resampling_method='sinc_interpolation')
        waveform = sample(waveform)
        # dither
        waveform = F.dither(waveform, noise_shaping=True)

    return waveform, sample_rate, utterance, speaker_id, utterance_id


def load_vctk_092_item(fileid: str,
                       path: str,
                       ext_audio: str,
                       ext_txt: str,
                       folder_audio: str,
                       folder_txt: str) -> Tuple[Tensor, int, str, str, str]:
    speaker_id, utterance_id, _ = fileid.split("_")

    # Read text
    #
    # After 30th Nov 2018 update, a new mic was used to record some audio files.
    # In the process all audio file names were changed to include the mic info.
    #
    # Old audio filename - p225_001
    # New audio filename - p225_001_mic1
    #
    # So, using string slicing to use the appropriate fileid for text files.
    file_txt = os.path.join(path, folder_txt, speaker_id, fileid[:-5] + ext_txt)

    with open(file_txt) as file_text:
        utterance = file_text.readlines()[0]

    # Read flac
    file_audio = os.path.join(path, folder_audio, speaker_id, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, utterance, speaker_id, utterance_id


class VCTK(Dataset):
    """
    Create a Dataset for VCTK. Each item is a tuple of the form:
    (waveform, sample_rate, utterance, speaker_id, utterance_id)

    Folder `p315` will be ignored due to the non-existent corresponding text files.
    For more information about the dataset visit: https://datashare.is.ed.ac.uk/handle/10283/3443
    """

    _folder_txt = "txt"
    _folder_audio = "wav48"
    _ext_txt = ".txt"
    _ext_audio = ".wav"
    _except_folder = "p315"

    def __init__(self,
                 root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False,
                 downsample: bool = False,
                 transform: Any = None,
                 target_transform: Any = None) -> None:

        if downsample:
            warnings.warn(
                "In the next version, transforms will not be part of the dataset. "
                "Please use `downsample=False` to enable this behavior now, "
                "and suppress this warning."
            )

        if transform is not None or target_transform is not None:
            warnings.warn(
                "In the next version, transforms will not be part of the dataset. "
                "Please remove the option `transform=True` and "
                "`target_transform=True` to suppress this warning."
            )

        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if download:
            raise RuntimeError(
                "This Dataset is no longer available. "
                "Please use `VCTK_092` class to download the latest version."
            )

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `VCTK_092` class "
                "with `download=True` to donwload the latest version."
            )

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        walker = filter(lambda w: self._except_folder not in w, walker)
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str]:
        fileid = self._walker[n]
        item = load_vctk_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_txt,
            self._folder_audio,
            self._folder_txt,
        )

        # TODO Upon deprecation, uncomment line below and remove following code
        # return item

        waveform, sample_rate, utterance, speaker_id, utterance_id = item
        if self.transform is not None:
            waveform = self.transform(waveform)
        if self.target_transform is not None:
            utterance = self.target_transform(utterance)
        return waveform, sample_rate, utterance, speaker_id, utterance_id

    def __len__(self) -> int:
        return len(self._walker)


class VCTK_092(Dataset):
    """
    Create a Dataset for VCTK 0.92, the latest version of the VCTK dataset.
    Each item is a tuple of the form: (waveform, sample_rate, utterance, speaker_id, utterance_id)
    Folder `p315` will be ignored due to the non-existent corresponding text files.
    For more information about the dataset visit: https://datashare.is.ed.ac.uk/handle/10283/3443
    """

    _folder_txt = "txt"
    _folder_audio = "wav48_silence_trimmed"
    _ext_txt = ".txt"
    _ext_audio = ".flac"
    _except_folder = "p315"

    def __init__(self,
                 root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE_092,
                 download: bool = False) -> None:

        archive = os.path.basename(url)

        # custom url may contain options like '?sequence=2&isAllowed=y'
        # Slicing it off to extract only 'VCTK-Corpus-0.92.zip'
        archive = os.path.join(root, archive[:20])

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive, self._path)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )

        walker = filter(lambda w: self._except_folder not in w, walker)
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str]:
        fileid = self._walker[n]
        item = load_vctk_092_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_txt,
            self._folder_audio,
            self._folder_txt,
        )

        return item

    def __len__(self) -> int:
        return len(self._walker)
