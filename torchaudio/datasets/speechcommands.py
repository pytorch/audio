import os
import glob

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    unicode_csv_reader
)

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"


def load_speechcommand_item(filepath, path):
    label, filename = os.path.split(filepath)
    filename, _ = os.path.splitext(filename)

    speaker_id = filename.split("_")[0]
    file_audio = os.path.join(path, filepath)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)
    return waveform, sample_rate, label, speaker_id


class SPEECHCOMMANDS(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label, speaker_id
    """

    def __init__(
            self,
            root,
            subset=None,
            url=URL,
            folder_in_archive=FOLDER_IN_ARCHIVE,
            download=False
    ):
        subsets = ["validation", "testing", "training"]
        suffix_subset_file = "_list.txt"

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive, self._path)

        walker = glob.glob(os.path.join(self._path, "**/*.wav"), recursive=True)
        self._walker = [os.path.relpath(sample_path, self._path) for sample_path in walker]

        if subset in subsets:
            if subset == "training":
                subsets.remove("training")
                for subset in subsets:
                    dataset_path = os.path.join(self._path, subset + suffix_subset_file)

                    with open(dataset_path, "r") as f:
                        walker = [sample[0] for sample in unicode_csv_reader(f)]
                        self._walker = list(set(self._walker).difference(walker))

            else:
                dataset_path = os.path.join(self._path, subset + suffix_subset_file)

                with open(dataset_path, "r") as f:
                    walker = [sample[0] for sample in unicode_csv_reader(f)]
                    self._walker = list(set(self._walker).intersection(walker))

    def __getitem__(self, n):
        fileid = self._walker[n]
        return load_speechcommand_item(fileid, self._path)

    def __len__(self):
        return len(self._walker)
