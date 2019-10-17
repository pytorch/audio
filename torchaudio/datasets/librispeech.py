import os

import torch.utils.data as data

import torchaudio
from torchaudio.datasets.utils import download, extract, shuffle, walk


def load_librispeech_item(fileid, path, ext_audio, ext_txt):

    speaker, chapter, utterance = fileid.split("-")

    file_text = speaker + "-" + chapter + ext_txt
    file_text = os.path.join(path, speaker, chapter, file_text)
    file_audio = speaker + "-" + chapter + "-" + utterance + ext_audio
    file_audio = os.path.join(path, speaker, chapter, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    for line in open(file_text):
        fileid_text, content = line.strip().split(" ", 1)
        if file_audio == fileid_text:
            break
    else:
        # Translation not found
        raise ValueError

    return {
        "speaker": speaker,
        "chapter": chapter,
        "utterance": utterance,
        "content": content,
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


class LIBRISPEECH(data.IterableDataset):

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(self, root, selection):

        selections = [
            "dev-clean",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]

        assert selection in selections

        ext_archive = ".tar.gz"
        base = "http://www.openslr.org/resources/12/"
        url = os.path.join(base, selection + ext_archive)
        folder_in_archive = os.path.join("LibriSpeech", selection)

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
            yield load_librispeech_item(
                fileid, self._path, self._ext_audio, self._ext_txt
            )
