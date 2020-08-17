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

RELEASE = "release1"  # Default release

_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "TEDLIUM_release1",
        "url": "http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz",
        "checksum": "ffd31f96d81a21bf4928eaf9bb0b0c2dea7a5247",
        "data_path": "",
        "subset": "train",
    },
    "release2": {
        "folder_in_archive": "TEDLIUM_release2",
        "url": "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz",
        "checksum": "5c8fb045246d1c64296f57b47aa7dc79d16b184f",
        "data_path": "",
        "subset": "train",
    },
    "release3": {
        "folder_in_archive": "TEDLIUM_release-3",
        "url": "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz",
        "checksum": "685d27c39c53217383d7933cc405a07048004127",
        "data_path": "data/",
        "subset": None,
    },
}


class TEDLIUM(Dataset):
    """
    Create a Dataset for Tedlium. Each item is a tuple of the form:
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
    """

    _ext_txt = ".stm"
    _ext_audio = ".sph"

    def __init__(self, root: str, release: str = RELEASE, subset: str = None, download: bool = False) -> None:

        if release in _RELEASE_CONFIGS.keys():
            folder_in_archive = _RELEASE_CONFIGS[release]["folder_in_archive"]
            url = _RELEASE_CONFIGS[release]["url"]
            subset = subset if subset else _RELEASE_CONFIGS[release]["subset"]
        else:
            # Raise warning
            raise RuntimeError(
                "The release {} does not match any of the supported tedlium releases{} ".format(
                    release, _RELEASE_CONFIGS.keys(),
                )
            )

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]

        self._path = os.path.join(root, folder_in_archive, _RELEASE_CONFIGS[release]["data_path"])
        if subset in ["train", "dev", "test"]:
            self._path = os.path.join(self._path, subset)
        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _RELEASE_CONFIGS[release]["checksum"]
                    download_url(url, root, hash_value=checksum)
                extract_archive(archive)

        walker = walk_files(self._path, suffix=self._ext_txt, prefix=False, remove_suffix=True)
        self._walker = list(walker)
        self._extended_walker = []
        for file in self._walker:
            stm_path = os.path.join(self._path, "stm", file + self._ext_txt)
            with open(stm_path) as f:
                l = len(f.readlines())
                self._extended_walker += [(file, line) for line in range(l)]

    def load_tedlium_item(self, fileid: str, line: int, path: str) -> Tuple[Tensor, int, str, int, int, int]:
        transcript_path = os.path.join(path, "stm", fileid)
        with open(transcript_path + self._ext_txt) as f:
            transcript = f.readlines()[line]
            talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(" ", 6)

        wave_path = os.path.join(path, "sph", fileid)
        waveform, sample_rate = self.load_audio(wave_path + self._ext_audio)
        # Calculate indexes for start time and endtime
        start_time = int(float(start_time) * sample_rate)
        end_time = int(float(end_time) * sample_rate)
        waveform = waveform[:, start_time:end_time]
        return (
            waveform,
            sample_rate,
            transcript,
            talk_id,
            speaker_id,
            identifier,
        )

    def load_audio(self, path: str) -> [Tensor, int]:
        return torchaudio.load(path)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        fileid, line = self._extended_walker[n]
        return self.load_tedlium_item(fileid, line, self._path)

    def __len__(self) -> int:
        return len(self._extended_walker)
