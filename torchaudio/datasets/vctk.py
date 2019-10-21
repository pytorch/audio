import os

import torchaudio
from torchaudio.datasets.utils import data, download_url, extract_archive, walk_files

DEFAULT_URL = "http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"


def load_vctk_item(fileid, path, ext_audio, ext_txt, folder_audio, folder_txt):
    speaker, utterance = fileid.split("_")

    # Read text
    file_txt = os.path.join(path, folder_txt, speaker, fileid + ext_txt)
    with open(file_txt) as file_text:
        content = file_text.readlines()[0]

    # Read wav
    file_audio = os.path.join(path, folder_audio, speaker, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return {
        "speaker": speaker,
        "utterance": utterance,
        "content": content,
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


class VCTK(data.Dataset):

    _folder_txt = "txt"
    _folder_audio = "wav48"
    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(self, root, url=DEFAULT_URL):

        folder_in_archive = "VCTK-Corpus/"

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                download_url(url, root)
            extract_archive(archive)

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        fileid = self._walker[n]
        return load_vctk_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_txt,
            self._folder_audio,
            self._folder_txt,
        )

    def __len__(self):
        return len(self._walker)
