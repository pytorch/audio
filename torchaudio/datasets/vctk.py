import os
import warnings
from pathlib import Path
from typing import Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset

import torchaudio
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)

URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
FOLDER_IN_ARCHIVE = "VCTK-Corpus"
_CHECKSUMS = {
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip": "8a6ba2946b36fcbef0212cad601f4bfa"
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


class VCTK(Dataset):
    """Create a Dataset for VCTK.

    Note:
        * **This dataset is no longer publicly available.** Please use :py:class:`VCTK_092`
        * Directory ``p315`` is ignored because there is no corresponding text files.
          For more information about the dataset visit: https://datashare.is.ed.ac.uk/handle/10283/3443

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): Not used as the dataset is no longer publicly available.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"VCTK-Corpus"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
            Giving ``download=True`` will result in error as the dataset is no longer
            publicly available.
        downsample (bool, optional): Not used.
    """

    _folder_txt = "txt"
    _folder_audio = "wav48"
    _ext_txt = ".txt"
    _ext_audio = ".wav"
    _except_folder = "p315"

    def __init__(self,
                 root: Union[str, Path],
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False,
                 downsample: bool = False) -> None:

        if downsample:
            warnings.warn(
                "In the next version, transforms will not be part of the dataset. "
                "Please use `downsample=False` to enable this behavior now, "
                "and suppress this warning."
            )

        self.downsample = downsample
        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

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

        walker = sorted(str(p.stem) for p in Path(self._path).glob('**/*' + self._ext_audio))
        walker = filter(lambda w: self._except_folder not in w, walker)
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, utterance_id)``
        """
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
        return waveform, sample_rate, utterance, speaker_id, utterance_id

    def __len__(self) -> int:
        return len(self._walker)


SampleType = Tuple[Tensor, int, str, str, str]


class VCTK_092(Dataset):
    """Create VCTK 0.92 Dataset

    Args:
        root (str): Root directory where the dataset's top level directory is found.
        mic_id (str): Microphone ID. Either ``"mic1"`` or ``"mic2"``. (default: ``"mic2"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        url (str, optional): The URL to download the dataset from.
            (default: ``"https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"``)
        audio_ext (str, optional): Custom audio extension if dataset is converted to non-default audio format.

    Note:
        * All the speeches from speaker ``p315`` will be skipped due to the lack of the corresponding text files.
        * All the speeches from ``p280`` will be skipped for ``mic_id="mic2"`` due to the lack of the audio files.
        * Some of the speeches from speaker ``p362`` will be skipped due to the lack of  the audio files.
        * See Also: https://datashare.is.ed.ac.uk/handle/10283/3443
    """

    def __init__(
            self,
            root: str,
            mic_id: str = "mic2",
            download: bool = False,
            url: str = URL,
            audio_ext=".flac",
    ):
        if mic_id not in ["mic1", "mic2"]:
            raise RuntimeError(
                f'`mic_id` has to be either "mic1" or "mic2". Found: {mic_id}'
            )

        archive = os.path.join(root, "VCTK-Corpus-0.92.zip")

        self._path = os.path.join(root, "VCTK-Corpus-0.92")
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48_silence_trimmed")
        self._mic_id = mic_id
        self._audio_ext = audio_ext

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
        self._speaker_ids = sorted(os.listdir(self._txt_dir))
        self._sample_ids = []

        """
        Due to some insufficient data complexity in the 0.92 version of this dataset,
        we start traversing the audio folder structure in accordance with the text folder.
        As some of the audio files are missing of either ``mic_1`` or ``mic_2`` but the
        text is present for the same, we first check for the existence of the audio file
        before adding it to the ``sample_ids`` list.

        Once the ``audio_ids`` are loaded into memory we can quickly access the list for
        different parameters required by the user.
        """
        for speaker_id in self._speaker_ids:
            if speaker_id == "p280" and mic_id == "mic2":
                continue
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            for utterance_file in sorted(
                    f for f in os.listdir(utterance_dir) if f.endswith(".txt")
            ):
                utterance_id = os.path.splitext(utterance_file)[0]
                audio_path_mic = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}_{mic_id}{self._audio_ext}",
                )
                if speaker_id == "p362" and not os.path.isfile(audio_path_mic):
                    continue
                self._sample_ids.append(utterance_id.split("_"))

    def _load_text(self, file_path) -> str:
        with open(file_path) as file_path:
            return file_path.readlines()[0]

    def _load_audio(self, file_path) -> Tuple[Tensor, int]:
        return torchaudio.load(file_path)

    def _load_sample(self, speaker_id: str, utterance_id: str, mic_id: str) -> SampleType:
        utterance_path = os.path.join(
            self._txt_dir, speaker_id, f"{speaker_id}_{utterance_id}.txt"
        )
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}_{mic_id}{self._audio_ext}",
        )

        # Reading text
        utterance = self._load_text(utterance_path)

        # Reading FLAC
        waveform, sample_rate = self._load_audio(audio_path)

        return (waveform, sample_rate, utterance, speaker_id, utterance_id)

    def __getitem__(self, n: int) -> SampleType:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, utterance, speaker_id, utterance_id)``
        """
        speaker_id, utterance_id = self._sample_ids[n]
        return self._load_sample(speaker_id, utterance_id, self._mic_id)

    def __len__(self) -> int:
        return len(self._sample_ids)
