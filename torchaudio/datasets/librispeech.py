import os

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader,
    walk_files,
)

URL = "train-clean-100"
FOLDER_IN_ARCHIVE = "LibriSpeech"


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
        "speaker_id": speaker,
        "chapter_id": chapter,
        "utterance_id": utterance,
        "utterance": content,
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


class LIBRISPEECH(Dataset):

    _ext_txt = ".trans.txt"
    _ext_audio = ".flac"

    def __init__(
        self, root, url=URL, folder_in_archive=FOLDER_IN_ARCHIVE, download=False
    ):

        if url in [
            "dev-clean",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ]:

            ext_archive = ".tar.gz"
            base_url = "http://www.openslr.org/resources/12/"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(".")[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
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
        return load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)

    def __len__(self):
        return len(self._walker)
