import csv
import os
from pathlib import Path
from typing import Tuple, Union

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio._internal import download_url_to_file
from torchaudio.datasets.utils import _extract_tar

URL = "aew"
FOLDER_IN_ARCHIVE = "ARCTIC"
_CHECKSUMS = {
    "http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2": "645cb33c0f0b2ce41384fdd8d3db2c3f5fc15c1e688baeb74d2e08cab18ab406",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_ahw_arctic.tar.bz2": "024664adeb892809d646a3efd043625b46b5bfa3e6189b3500b2d0d59dfab06c",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_aup_arctic.tar.bz2": "2c55bc3050caa996758869126ad10cf42e1441212111db034b3a45189c18b6fc",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_awb_arctic.tar.bz2": "d74a950c9739a65f7bfc4dfa6187f2730fa03de5b8eb3f2da97a51b74df64d3c",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_axb_arctic.tar.bz2": "dd65c3d2907d1ee52f86e44f578319159e60f4bf722a9142be01161d84e330ff",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2": "26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_clb_arctic.tar.bz2": "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_eey_arctic.tar.bz2": "8a0ee4e5acbd4b2f61a4fb947c1730ab3adcc9dc50b195981d99391d29928e8a",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_fem_arctic.tar.bz2": "3fcff629412b57233589cdb058f730594a62c4f3a75c20de14afe06621ef45e2",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_gka_arctic.tar.bz2": "dc82e7967cbd5eddbed33074b0699128dbd4482b41711916d58103707e38c67f",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_jmk_arctic.tar.bz2": "3a37c0e1dfc91e734fdbc88b562d9e2ebca621772402cdc693bbc9b09b211d73",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_ksp_arctic.tar.bz2": "8029cafce8296f9bed3022c44ef1e7953332b6bf6943c14b929f468122532717",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_ljm_arctic.tar.bz2": "b23993765cbf2b9e7bbc3c85b6c56eaf292ac81ee4bb887b638a24d104f921a0",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_lnh_arctic.tar.bz2": "4faf34d71aa7112813252fb20c5433e2fdd9a9de55a00701ffcbf05f24a5991a",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_rms_arctic.tar.bz2": "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_rxr_arctic.tar.bz2": "1fa4271c393e5998d200e56c102ff46fcfea169aaa2148ad9e9469616fbfdd9b",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_slp_arctic.tar.bz2": "54345ed55e45c23d419e9a823eef427f1cc93c83a710735ec667d068c916abf1",  # noqa: E501
    "http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2": "7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea",  # noqa: E501
}


def load_cmuarctic_item(line: str, path: str, folder_audio: str, ext_audio: str) -> Tuple[Tensor, int, str, str]:

    utterance_id, transcript = line[0].strip().split(" ", 2)[1:]

    # Remove space, double quote, and single parenthesis from transcript
    transcript = transcript[1:-3]

    file_audio = os.path.join(path, folder_audio, utterance_id + ext_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    return (waveform, sample_rate, transcript, utterance_id.split("_")[1])


class CMUARCTIC(Dataset):
    """*CMU ARCTIC* :cite:`Kominek03cmuarctic` dataset.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional):
            The URL to download the dataset from or the type of the dataset to download.
            (default: ``"aew"``)
            Allowed type values are ``"aew"``, ``"ahw"``, ``"aup"``, ``"awb"``, ``"axb"``, ``"bdl"``,
            ``"clb"``, ``"eey"``, ``"fem"``, ``"gka"``, ``"jmk"``, ``"ksp"``, ``"ljm"``, ``"lnh"``,
            ``"rms"``, ``"rxr"``, ``"slp"`` or ``"slt"``.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"ARCTIC"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
    """

    _file_text = "txt.done.data"
    _folder_text = "etc"
    _ext_audio = ".wav"
    _folder_audio = "wav"

    def __init__(
        self, root: Union[str, Path], url: str = URL, folder_in_archive: str = FOLDER_IN_ARCHIVE, download: bool = False
    ) -> None:

        if url in [
            "aew",
            "ahw",
            "aup",
            "awb",
            "axb",
            "bdl",
            "clb",
            "eey",
            "fem",
            "gka",
            "jmk",
            "ksp",
            "ljm",
            "lnh",
            "rms",
            "rxr",
            "slp",
            "slt",
        ]:

            url = "cmu_us_" + url + "_arctic"
            ext_archive = ".tar.bz2"
            base_url = "http://www.festvox.org/cmu_arctic/packed/"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        basename = os.path.basename(url)
        root = os.path.join(root, folder_in_archive)
        if not os.path.isdir(root):
            os.mkdir(root)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]

        self._path = os.path.join(root, basename)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                _extract_tar(archive)
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )
        self._text = os.path.join(self._path, self._folder_text, self._file_text)

        with open(self._text, "r") as text:
            walker = csv.reader(text, delimiter="\n")
            self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            str:
                Utterance ID
        """
        line = self._walker[n]
        return load_cmuarctic_item(line, self._path, self._folder_audio, self._ext_audio)

    def __len__(self) -> int:
        return len(self._walker)
