import os
import warnings
from typing import Any, Tuple
from collections import namedtuple

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

Sample = namedtuple('Sample', ['waveform', 'sample_rate', 'utterance', 'speaker_id', 'utterance_id'])


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
    _mic_1 = "mic1"
    _mic_2 = "mic2"

    def __init__(self,
                 root: str,
                 url: str = URL,
                 download: bool = False) -> None:

        # Custom url may contain options like '?sequence=2&isAllowed=y'
        # Slicing it off to extract only 'VCTK-Corpus-0.92.zip'
        archive = os.path.join(root, os.path.basename(url)[:20])

        self._path = os.path.join(root, FOLDER_IN_ARCHIVE_092)
        self._txt_dir = os.path.join(self._path, self._folder_txt)
        self._audio_dir = os.path.join(self._path, self._folder_audio)

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

        # Extracting speaker IDs from the folder structure
        self._speaker_ids = sorted(d for d in os.listdir(self._txt_dir)
                                   if os.path.isdir(os.path.join(self._txt_dir, d)))
        self._audio_ids = []

        """
        Due to some insufficient data complexity in the 0.92 version of this dataset,
        we start traversing the audio folder structure in accordance with the text folder.
        As some of the audio files are missing of either ``mic_1`` or ``mic_2`` but the 
        text is present for the same, we first check for the existence of the audio file
        before adding it to the ``audio_ids`` list.

        Once the ``audio_ids`` are loaded into memory we can quickly access the list for
        different parameters required by the user.
        """
        for speaker_id in self._speaker_ids:
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(f for f in os.listdir(utterance_dir) if f.endswith(self._ext_txt)):
                utterance_id = os.path.splitext(utterance_file)[0]
                audio_path_mic1 = os.path.join(self._audio_dir, speaker_id,
                                               f'{utterance_id}_{self._mic_1}{self._ext_audio}')
                audio_path_mic2 = os.path.join(self._audio_dir, speaker_id,
                                               f'{utterance_id}_{self._mic_2}{self._ext_audio}')
                if os.path.exists(audio_path_mic1):
                    self._audio_ids.append(f'{utterance_id}_{self._mic_1}')
                if os.path.exists(audio_path_mic2):
                    self._audio_ids.append(f'{utterance_id}_{self._mic_2}')

    def load_text(self, file_path) -> str:
        with open(file_path) as file_path:
            return file_path.readlines()[0]

    def load_audio(self, file_path) -> Tuple[Tensor, int]:
        return torchaudio.load(file_path)

    def load_sample(self, audio_id: str) -> Sample:
        speaker_id, utterance_id, _ = audio_id.split("_")
        utterance_path = os.path.join(self._txt_dir, speaker_id, f'{speaker_id}_{utterance_id}{self._ext_txt}')
        audio_path = os.path.join(self._audio_dir, speaker_id, f'{audio_id}{self._ext_audio}')

        # Read text
        utterance = self.load_text(utterance_path)

        # Read FLAC
        waveform, sample_rate = self.load_audio(audio_path)

        return Sample(waveform, sample_rate, utterance, speaker_id, utterance_id)

    def __getitem__(self, n: int) -> Sample:
        audio_id = self._audio_ids[n]
        return self.load_sample(audio_id)

    def __len__(self) -> int:
        return len(self._audio_ids)
