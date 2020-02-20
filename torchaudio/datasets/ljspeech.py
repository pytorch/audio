import os

import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive, walk_files
from torch.utils.data import Dataset

URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
FOLDER_IN_ARCHIVE = "wavs"


class LJSPEECH(Dataset):
    """
    Create a Dataset for LJSpeech-1.1. Each item is a tuple of the form:
    waveform, sample_rate, transcript, normalized_transcript
    """

    _ext_audio = ".wav"
    _ext_archive = '.tar.bz2'

    def __init__(
            self, root, url=URL, folder_in_archive=FOLDER_IN_ARCHIVE, download=False
    ):

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.split(self._ext_archive)[0]
        folder_in_archive = os.path.join(basename, folder_in_archive)

        self._path = os.path.join(root, folder_in_archive)
        self._metadata_path = os.path.join(root, basename, 'metadata.csv')

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive)

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)
        self._metadata = self._load_metadata()

    def _load_metadata(self):
        metadata = dict()
        with open(self._metadata_path, 'r') as f:
            for row in f:
                fileid, transcript, normalized_transcript = row.strip().split('|')
                metadata[fileid] = {'transcript': transcript,
                                    'normalized_transcript': normalized_transcript}
        return metadata

    def load_ljspeech_item(self, fileid, path, ext_audio):

        fileid_audio = fileid + ext_audio
        fileid_audio = os.path.join(path, fileid_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(fileid_audio)

        # Load transcription
        fileid_metadata = self._metadata.get(fileid, None)
        if fileid_metadata is None:
            raise ValueError("Transcription not found for " + fileid_audio)

        return (
            waveform,
            sample_rate,
            fileid_metadata['transcript'],
            fileid_metadata['normalized_transcript'],
        )

    def __getitem__(self, n):
        fileid = self._walker[n]
        return self.load_ljspeech_item(fileid, self._path, self._ext_audio)

    def __len__(self):
        return len(self._walker)
