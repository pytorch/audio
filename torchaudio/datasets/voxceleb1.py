import os
from pathlib import Path
from typing import List, Tuple, Union

from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_zip, _load_waveform


SAMPLE_RATE = 16000
_ARCHIVE_CONFIGS = {
    "dev": {
        "archive_name": "vox1_dev_wav.zip",
        "urls": [
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa",
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab",
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac",
            "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad",
        ],
        "checksums": [
            "21ec6ca843659ebc2fdbe04b530baa4f191ad4b0971912672d92c158f32226a0",
            "311d21e0c8cbf33573a4fce6c80e5a279d80736274b381c394319fc557159a04",
            "92b64465f2b2a3dc0e4196ae8dd6828cbe9ddd1f089419a11e4cbfe2e1750df0",
            "00e6190c770b27f27d2a3dd26ee15596b17066b715ac111906861a7d09a211a5",
        ],
    },
    "test": {
        "archive_name": "vox1_test_wav.zip",
        "url": "https://thor.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip",
        "checksum": "8de57f347fe22b2c24526e9f444f689ecf5096fc2a92018cf420ff6b5b15eaea",
    },
}
_IDEN_SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
_VERI_TEST_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt"


def _download_extract_wavs(root: str):
    for archive in ["dev", "test"]:
        archive_name = _ARCHIVE_CONFIGS[archive]["archive_name"]
        archive_path = os.path.join(root, archive_name)
        # The zip file of dev data is splited to 4 chunks.
        # Download and combine them into one file before extraction.
        if archive == "dev":
            urls = _ARCHIVE_CONFIGS[archive]["urls"]
            checksums = _ARCHIVE_CONFIGS[archive]["checksums"]
            with open(archive_path, "wb") as f:
                for url, checksum in zip(urls, checksums):
                    file_path = os.path.join(root, os.path.basename(url))
                    download_url_to_file(url, file_path, hash_prefix=checksum)
                    with open(file_path, "rb") as f_split:
                        f.write(f_split.read())
        else:
            url = _ARCHIVE_CONFIGS[archive]["url"]
            checksum = _ARCHIVE_CONFIGS[archive]["checksum"]
            download_url_to_file(url, archive_path, hash_prefix=checksum)
        _extract_zip(archive_path)


def _get_flist(root: str, file_path: str, subset: str) -> List[str]:
    f_list = []
    if subset == "train":
        index = 1
    elif subset == "dev":
        index = 2
    else:
        index = 3
    with open(file_path, "r") as f:
        for line in f:
            id, path = line.split()
            if int(id) == index:
                f_list.append(path)
    return sorted(f_list)


def _get_paired_flist(root: str, veri_test_path: str):
    f_list = []
    with open(veri_test_path, "r") as f:
        for line in f:
            label, path1, path2 = line.split()
            f_list.append((label, path1, path2))
    return f_list


def _get_file_id(file_path: str, _ext_audio: str):
    speaker_id, youtube_id, utterance_id = file_path.split("/")[-3:]
    utterance_id = utterance_id.replace(_ext_audio, "")
    file_id = "-".join([speaker_id, youtube_id, utterance_id])
    return file_id


class VoxCeleb1(Dataset):
    """*VoxCeleb1* :cite:`nagrani2017voxceleb` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (Default: ``False``).
    """

    _ext_audio = ".wav"

    def __init__(self, root: Union[str, Path], download: bool = False) -> None:
        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._path = os.path.join(root, "wav")
        if not os.path.isdir(self._path):
            if not download:
                raise RuntimeError(
                    f"Dataset not found at {self._path}. Please set `download=True` to download the dataset."
                )
            _download_extract_wavs(root)

    def get_metadata(self, n: int):
        raise NotImplementedError

    def __getitem__(self, n: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class VoxCeleb1Identification(VoxCeleb1):
    """*VoxCeleb1* :cite:`nagrani2017voxceleb` dataset for speaker identification task.

    Each data sample contains the waveform, sample rate, speaker id, and the file id.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        subset (str, optional): Subset of the dataset to use. Options: ["train", "dev", "test"]. (Default: ``"train"``)
        meta_url (str, optional): The url of meta file that contains the list of subset labels and file paths.
            The format of each row is ``subset file_path". For example: ``1 id10006/nLEBBc9oIFs/00003.wav``.
            ``1``, ``2``, ``3`` mean ``train``, ``dev``, and ``test`` subest, respectively.
            (Default: ``"https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (Default: ``False``).

    Note:
        The file structure of `VoxCeleb1Identification` dataset is as follows:

        └─ root/

         └─ wav/

         └─ speaker_id folders

        Users who pre-downloaded the ``"vox1_dev_wav.zip"`` and ``"vox1_test_wav.zip"`` files need to move
        the extracted files into the same ``root`` directory.
    """

    def __init__(
        self, root: Union[str, Path], subset: str = "train", meta_url: str = _IDEN_SPLIT_URL, download: bool = False
    ) -> None:
        super().__init__(root, download)
        if subset not in ["train", "dev", "test"]:
            raise ValueError("`subset` must be one of ['train', 'dev', 'test']")
        # download the iden_split.txt to get the train, dev, test lists.
        meta_list_path = os.path.join(root, os.path.basename(meta_url))
        if not os.path.exists(meta_list_path):
            download_url_to_file(meta_url, meta_list_path)
        self._flist = _get_flist(self._path, meta_list_path, subset)

    def get_metadata(self, n: int) -> Tuple[str, int, int, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Sample rate
            int:
                Speaker ID
            str:
                File ID
        """
        file_path = self._flist[n]
        file_id = _get_file_id(file_path, self._ext_audio)
        speaker_id = file_id.split("-")[0]
        speaker_id = int(speaker_id[3:])
        return file_path, SAMPLE_RATE, speaker_id, file_id

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            int:
                Speaker ID
            str:
                File ID
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        return (waveform,) + metadata[1:]

    def __len__(self) -> int:
        return len(self._flist)


class VoxCeleb1Verification(VoxCeleb1):
    """*VoxCeleb1* :cite:`nagrani2017voxceleb` dataset for speaker verification task.

    Each data sample contains a pair of waveforms, sample rate, the label indicating if they are
    from the same speaker, and the file ids.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        meta_url (str, optional): The url of meta file that contains a list of utterance pairs
            and the corresponding labels. The format of each row is ``label file_path1 file_path2".
            For example: ``1 id10270/x6uYqmx31kE/00001.wav id10270/8jEAjG6SegY/00008.wav``.
            ``1`` means the two utterances are from the same speaker, ``0`` means not.
            (Default: ``"https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (Default: ``False``).

    Note:
        The file structure of `VoxCeleb1Verification` dataset is as follows:

        └─ root/

         └─ wav/

         └─ speaker_id folders

        Users who pre-downloaded the ``"vox1_dev_wav.zip"`` and ``"vox1_test_wav.zip"`` files need to move
        the extracted files into the same ``root`` directory.
    """

    def __init__(self, root: Union[str, Path], meta_url: str = _VERI_TEST_URL, download: bool = False) -> None:
        super().__init__(root, download)
        # download the veri_test.txt to get the list of training pairs and labels.
        meta_list_path = os.path.join(root, os.path.basename(meta_url))
        if not os.path.exists(meta_list_path):
            download_url_to_file(meta_url, meta_list_path)
        self._flist = _get_paired_flist(self._path, meta_list_path)

    def get_metadata(self, n: int) -> Tuple[str, str, int, int, str, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepaths instead of waveforms,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample

        Returns:
            Tuple of the following items;

            str:
                Path to audio file of speaker 1
            str:
                Path to audio file of speaker 2
            int:
                Sample rate
            int:
                Label
            str:
                File ID of speaker 1
            str:
                File ID of speaker 2
        """
        label, file_path_spk1, file_path_spk2 = self._flist[n]
        label = int(label)
        file_id_spk1 = _get_file_id(file_path_spk1, self._ext_audio)
        file_id_spk2 = _get_file_id(file_path_spk2, self._ext_audio)
        return file_path_spk1, file_path_spk2, SAMPLE_RATE, label, file_id_spk1, file_id_spk2

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, int, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform of speaker 1
            Tensor:
                Waveform of speaker 2
            int:
                Sample rate
            int:
                Label
            str:
                File ID of speaker 1
            str:
                File ID of speaker 2
        """
        metadata = self.get_metadata(n)
        waveform_spk1 = _load_waveform(self._path, metadata[0], metadata[2])
        waveform_spk2 = _load_waveform(self._path, metadata[1], metadata[2])
        return (waveform_spk1, waveform_spk2) + metadata[2:]

    def __len__(self) -> int:
        return len(self._flist)
