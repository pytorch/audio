import os
from warnings import warn

import torch.utils.data as data
import torchaudio
from torchaudio.datasets.utils import download, extract, shuffle, walk


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
        folder = fileid.split("_")[0]
        txt_file = os.path.join(path, txt_folder, folder, fileid + txt_extension)
        audio_file = os.path.join(path, audio_folder, folder, fileid + audio_extension)

        try:
            with open(txt_file) as txt_file:
                content = txt_file.readlines()[0]
        except FileNotFoundError:
            warn("Translation not found for {}".format(audio_file))
            # warn("File not found: {}".format(txt_file))
            continue

        waveform, sample_rate = torchaudio.load(audio_file)

        yield {
            "id": fileid,
            "content": content,
            "waveform": waveform,
            "sample_rate": sample_rate,
        }


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
        folder = fileid.split("_")[0]

        # Read text
        file_txt = os.path.join(
            self._path, self._folder_txt, folder, fileid + self._ext_txt
        )
        with open(file_txt) as txt_file:
            content = txt_file.readlines()[0]

        # Read wav
        file_audio = os.path.join(
            self._path, self._folder_audio, folder, fileid + self._ext_audio
        )
        waveform, sample_rate = torchaudio.load(file_audio)

        return {
            "id": fileid,
            "content": content,
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

    def __len__(self):
        return len(self._list)
