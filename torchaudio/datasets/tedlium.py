import os
from typing import Tuple

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)


_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "TEDLIUM_release1",
        "url": "http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz",
        "checksum": "30301975fd8c5cac4040c261c0852f57cfa8adbbad2ce78e77e4986957445f27",
        "data_path": "",
        "subset": "train",
        "supported_subsets": ["train", "test", "dev"],
        "dict": "TEDLIUM.150K.dic",
    },
    "release2": {
        "folder_in_archive": "TEDLIUM_release2",
        "url": "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz",
        "checksum": "93281b5fcaaae5c88671c9d000b443cb3c7ea3499ad12010b3934ca41a7b9c58",
        "data_path": "",
        "subset": "train",
        "supported_subsets": ["train", "test", "dev"],
        "dict": "TEDLIUM.152k.dic",
    },
    "release3": {
        "folder_in_archive": "TEDLIUM_release-3",
        "url": "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz",
        "checksum": "ad1e454d14d1ad550bc2564c462d87c7a7ec83d4dc2b9210f22ab4973b9eccdb",
        "data_path": "data/",
        "subset": None,
        "supported_subsets": [None],
        "dict": "TEDLIUM.152k.dic",
    },
}


class TEDLIUM(Dataset):
    """
    Create a Dataset for Tedlium. It supports releases 1,2 and 3, each item is a list containings:
    [waveform, sample_rate, transcript, talk_id, speaker_id, identifier].

    Constructor arguments:

    Args:
        root (str): Path containing dataset or target path where its downloaded if needed
        release (str, optional): TEDLIUM identifier (release1,release2,release3). Defaults to RELEASE.
        subset (str, optional): train/dev/test for releases 1&2, None for release3. Defaults to Train/None
        download (bool, optional): Download dataset in case is not founded in root path. Defaults to False.
        audio_ext (str, optional): Overwrite audio extension when loading items. Defaults to ".sph".

    Special functions:

    _load_tedlium_item: Loads a TEDLIUM dataset sample given a file name and corresponding sentence name

    _load_audio: Default load function used in TEDLIUM dataset, you can overwrite this function to customize
                 functionality and load individual sentences from a full ted audio talk file

    get_phoneme_dict: Returns the phoneme dictionary of a TEDLIUM release

    """

    def __init__(
        self, root: str, release: str = "release1", subset: str = None, download: bool = False, audio_ext=".sph"
    ) -> None:
        """Constructor for TEDLIUM dataset.

        Args:
            root (str): Path containing dataset or target path where its downloaded if needed
            release (str, optional): TEDLIUM identifier (release1,release2,release3). Defaults to RELEASE.
            subset (str, optional): train/dev/test for releases 1&2, None for release3. Defaults to Train/None
            download (bool, optional): Download dataset in case is not founded in root path. Defaults to False.
            audio_ext (str, optional): Overwrite audio extension when loading items. Defaults to ".sph".

        Raises:
            RuntimeError: If release identifier does not match any supported release,
        """
        self._ext_audio = audio_ext
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
        if subset not in _RELEASE_CONFIGS[release]["supported_subsets"]:
            # Raise warning
            raise RuntimeError(
                "The subset {} does not match any of the supported tedlium subsets{} ".format(
                    subset, _RELEASE_CONFIGS[release]["supported_subsets"],
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

        # Create list for all samples
        self._filelist = []
        stm_path = os.path.join(self._path, "stm")
        for file in sorted(os.listdir(stm_path)):
            if file.endswith(".stm"):
                stm_path = os.path.join(self._path, "stm", file)
                with open(stm_path) as f:
                    l = len(f.readlines())
                    file = file.replace(".stm", "")
                    self._filelist.extend((file, line) for line in range(l))
        # Create dict path for later read
        self._dict_path = os.path.join(root, folder_in_archive, _RELEASE_CONFIGS[release]["dict"])
        self._phoneme_dict = None

    def _load_tedlium_item(self, fileid: str, line: int, path: str) -> Tuple[Tensor, int, str, int, int, int]:
        """Loads a TEDLIUM dataset sample given a file name and corresponding sentence name.

        Args:
            fileid (str): File id to identify both text and audio files corresponding to the sample
            line (int): Line identifier for the sample inside the text file
            path (str): Dataset root path

        Returns:
            Tedlium_item: A namedTuple containing [waveform, sample_rate, transcript, talk_id, speaker_id, identifier]
        """
        transcript_path = os.path.join(path, "stm", fileid)
        with open(transcript_path + ".stm") as f:
            transcript = f.readlines()[line]
            talk_id, _, speaker_id, start_time, end_time, identifier, transcript = transcript.split(" ", 6)

        wave_path = os.path.join(path, "sph", fileid)
        waveform, sample_rate = self._load_audio(wave_path + self._ext_audio, start_time=start_time, end_time=end_time)

        return (waveform, sample_rate, transcript, talk_id, speaker_id, identifier)

    def _load_audio(self, path: str, start_time: float, end_time: float, sample_rate: int = 16000) -> [Tensor, int]:
        """Default load function used in TEDLIUM dataset, you can overwrite this function to customize functionality
        and load individual sentences from a full ted audio talk file.

        Args:
            path (str): Path to audio file
            start_time (int, optional): Time in seconds where the sample sentence stars
            end_time (int, optional): Time in seconds where the sample sentence finishes

        Returns:
            [Tensor, int]: Audio tensor representation and sample rate
        """
        start_time = int(float(start_time) * 16000)
        end_time = int(float(end_time) * 16000)
        if torchaudio.get_audio_backend() == "sox_io":
            return torchaudio.load(path, frame_offset=start_time, num_frames=end_time - start_time)
        return torchaudio.load(path)[:, start_time:end_time]

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        """TEDLIUM dataset custom function overwritting default loadbehaviour
        Loads a TEDLIUM sample given a index N.

        Args:
            n (int): Index of sample to be loaded

        Returns:
            Tedlium_item: A namedTuple containing [waveform, sample_rate, transcript, talk_id, speaker_id, identifier]
        """
        fileid, line = self._filelist[n]
        return self._load_tedlium_item(fileid, line, self._path)

    def __len__(self) -> int:
        """TEDLIUM dataset custom function overwritting len default behaviour.

        Returns:
            int: TEDLIUM dataset length
        """
        return len(self._filelist)

    @property
    def phoneme_dict(self):
        """Returns the phoneme dictionary of a TEDLIUM release.

        Returns:
            dictionary: Phoneme dictionary for the current tedlium release
        """
        # Read phoneme dictionary
        if not self._phoneme_dict:
            self._phoneme_dict = {}
            with open(self.dict_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    content = line.strip().split(maxsplit=1)
                    self._phoneme_dict[content[0]] = tuple(content[1:])  # content[1:] can be empty list
        return self._phoneme_dict.copy()
