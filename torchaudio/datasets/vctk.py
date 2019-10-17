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


def load_vctk(fileids):
    """
    Load data corresponding to each VCTK fileids.

    Input: path, file name identifying a row of data
    Output: dictionary with id, content, waveform, sample_rate
    """

    txt_folder = "txt"
    txt_extension = ".txt"

    audio_folder = "wav48"
    audio_extension = ".wav"

    for path, fileid in fileids:

        fileid = os.path.basename(fileid).split(".")[0]
        try:
            yield load_vctk_item(
                fileid, path, audio_extension, txt_extension, audio_folder, txt_folder
            )
        except FileNotFoundError:
            warn("Translation not found for {}".format(fileid))
            continue


def VCTK(root):
    """
    Create a generator for VCTK.
    """

    url = [
        (
            "http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz",
            "VCTK-Corpus/",
        )
    ]

    path = download(url, root_path=root)
    path = extract(path)
    path = walk(path, extension=".wav")
    # path = shuffle(path)
    # path, l = generator_length(path)
    return load_vctk(path)


class VCTK2(data.Dataset):

    _folder_txt = "txt"
    _folder_audio = "wav48"
    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(self, root):

        url = "http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"
        folder_in_archive = "VCTK-Corpus/"

        # torchaudio.datasets.utils.download_url(_url, root)

        filename = os.path.basename(url)
        filename = os.path.join(root, filename)
        # torchaudio.datasets.utils.extract_archive(filename)

        self._path = os.path.join(root, folder_in_archive)

        self._list = torchaudio.datasets.utils.list_files_recursively(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )

    def __getitem__(self, n):
        fileid = self._list[n]
        return load_vctk_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_txt,
            self._folder_audio,
            self._folder_txt,
        )

    def __len__(self):
        return len(self._list)
