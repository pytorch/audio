import os
from warnings import warn

import torch.utils.data as data

import torchaudio
from torchaudio.datasets.utils import download, extract, shuffle, walk


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


class VCTK(data.IterableDataset):

    _folder_txt = "txt"
    _folder_audio = "wav48"
    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(self, root):

        url = "http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"
        folder_in_archive = "VCTK-Corpus/"

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if not os.path.isdir(self._path):
            if not os.path.isfile(archive):
                torchaudio.datasets.utils.download_url(_url, root)
            torchaudio.datasets.utils.extract_archive(archive)

        walker = torchaudio.datasets.utils.walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = walker

    def __iter__(self):
        for fileid in self._walker:
            yield load_vctk_item(
                fileid,
                self._path,
                self._ext_audio,
                self._ext_txt,
                self._folder_audio,
                self._folder_txt,
            )
