import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files,
)

URL = "cmu_us_aew_arctic"
FOLDER_IN_ARCHIVE = "ARCTIC"
_CHECKSUMS = {
    "http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2":
    "4382b116efcc8339c37e01253cb56295",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ahw_arctic.tar.bz2":
    "b072d6e961e3f36a2473042d097d6da9",
    "http://festvox.org/cmu_arctic/packed/cmu_us_aup_arctic.tar.bz2":
    "5301c7aee8919d2abd632e2667adfa7f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_awb_arctic.tar.bz2":
    "280fdff1e9857119d9a2c57b50e12db7",
    "http://festvox.org/cmu_arctic/packed/cmu_us_axb_arctic.tar.bz2":
    "5e21cb26c6529c533df1d02ccde5a186",
    "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2":
    "b2c3e558f656af2e0a65da0ac0c3377a",
    "http://festvox.org/cmu_arctic/packed/cmu_us_clb_arctic.tar.bz2":
    "3957c503748e3ce17a3b73c1b9861fb0",
    "http://festvox.org/cmu_arctic/packed/cmu_us_eey_arctic.tar.bz2":
    "59708e932d27664f9eda3e8e6859969b",
    "http://festvox.org/cmu_arctic/packed/cmu_us_fem_arctic.tar.bz2":
    "dba4f992ff023347c07c304bf72f4c73",
    "http://festvox.org/cmu_arctic/packed/cmu_us_gka_arctic.tar.bz2":
    "24a876ea7335c1b0ff21460e1241340f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_jmk_arctic.tar.bz2":
    "afb69d95f02350537e8a28df5ab6004b",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ksp_arctic.tar.bz2":
    "4ce5b3b91a0a54b6b685b1b05aa0b3be",
    "http://festvox.org/cmu_arctic/packed/cmu_us_ljm_arctic.tar.bz2":
    "6f45a3b2c86a4ed0465b353be291f77d",
    "http://festvox.org/cmu_arctic/packed/cmu_us_lnh_arctic.tar.bz2":
    "c6a15abad5c14d27f4ee856502f0232f",
    "http://festvox.org/cmu_arctic/packed/cmu_us_rms_arctic.tar.bz2":
    "71072c983df1e590d9e9519e2a621f6e",
    "http://festvox.org/cmu_arctic/packed/cmu_us_rxr_arctic.tar.bz2":
    "3771ff03a2f5b5c3b53aa0a68b9ad0d5",
    "http://festvox.org/cmu_arctic/packed/cmu_us_slp_arctic.tar.bz2":
    "9cbf984a832ea01b5058ba9a96862850",
    "http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic.tar.bz2":
    "959eecb2cbbc4ac304c6b92269380c81",
}


def load_cmu_arctic_item(fileid: str,
                         path: str,
                         ext_audio: str,
                         ext_txt: str) -> Tuple[Tensor, int, str, str]:
    utterance_id = fileid

    file_text = os.path.join(path, "etc", ext_txt)

    file_audio = os.path.join(path, "wav", utterance_id + ext_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            file_id, utterance = line.strip().split(" ", 2)[1:]
            if fileid == file_id:
                utterance = utterance[1:-3]
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid)

    return (
        waveform,
        sample_rate,
        utterance,
        utterance_id.split("_")[1]
    )


class CMU_ARCTIC(Dataset):
    """
    Create a Dataset for CMU_Arctic. Each item is a tuple of the form:
    waveform, sample_rate, utterance, utterance_id
    """

    _ext_txt = "txt.done.data"
    _ext_audio = ".wav"

    def __init__(self,
                 root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        if url in [
            "cmu_us_aew_arctic",
            "cmu_us_ahw_arctic",
            "cmu_us_aup_arctic",
            "cmu_us_awb_arctic",
            "cmu_us_axb_arctic",
            "cmu_us_bdl_arctic",
            "cmu_us_clb_arctic",
            "cmu_us_eey_arctic",
            "cmu_us_fem_arctic",
            "cmu_us_gka_arctic",
            "cmu_us_jmk_arctic",
            "cmu_us_ksp_arctic",
            "cmu_us_ljm_arctic",
            "cmu_us_lnh_arctic",
            "cmu_us_rms_arctic",
            "cmu_us_rxr_arctic",
            "cmu_us_slp_arctic",
            "cmu_us_slt_arctic"
        ]:

            ext_archive = ".tar.bz2"
            base_url = "http://www.festvox.org/cmu_arctic/packed/"

            url = os.path.join(base_url, url + ext_archive)

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
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str]:
        fileid = self._walker[n]
        return load_cmu_arctic_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self) -> int:
        return len(self._walker)
