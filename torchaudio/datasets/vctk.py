import os
import warnings

import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive, walk_files

URL = "http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"
FOLDER_IN_ARCHIVE = "VCTK-Corpus"


def load_vctk_item(
    fileid, path, ext_audio, ext_txt, folder_audio, folder_txt, downsample=False
):
    speaker, utterance = fileid.split("_")

    # Read text
    file_txt = os.path.join(path, folder_txt, speaker, fileid + ext_txt)
    with open(file_txt) as file_text:
        content = file_text.readlines()[0]

    # Read wav
    file_audio = os.path.join(path, folder_audio, speaker, fileid + ext_audio)
    if downsample:
        # Legacy
        E = torchaudio.sox_effects.SoxEffectsChain()
        E.set_input_file(file_audio)
        E.append_effect_to_chain("gain", ["-h"])
        E.append_effect_to_chain("channels", [1])
        E.append_effect_to_chain("rate", [16000])
        E.append_effect_to_chain("gain", ["-rh"])
        E.append_effect_to_chain("dither", ["-s"])
        waveform, sample_rate = E.sox_build_flow_effects()
    else:
        waveform, sample_rate = torchaudio.load(file_audio)

    return {
        "speaker_id": speaker,
        "utterance_id": utterance,
        "utterance": content,
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


class VCTK(Dataset):

    _folder_txt = "txt"
    _folder_audio = "wav48"
    _ext_txt = ".txt"
    _ext_audio = ".wav"

    def __init__(
        self,
        root,
        url=URL,
        folder_in_archive=FOLDER_IN_ARCHIVE,
        download=False,
        downsample=False,
        transform=None,
        target_transform=None,
        return_dict=False,
    ):

        if not return_dict:
            warnings.warn(
                "In the next version, the item returned will be a dictionary. "
                "Please use `return_dict=True` to enable this behavior now, "
                "and suppress this warning.",
                DeprecationWarning,
            )

        if downsample:
            warnings.warn(
                "In the next version, transforms will not be part of the dataset. "
                "Please use `downsample=False` to enable this behavior now, ",
                "and suppress this warning.",
                DeprecationWarning,
            )

        if transform is not None or target_transform is not None:
            warnings.warn(
                "In the next version, transforms will not be part of the dataset. "
                "Please remove the option `transform=True` and "
                "`target_transform=True` to suppress this warning.",
                DeprecationWarning,
            )

        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self.return_dict = return_dict

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        self._walker = list(walker)

    def __getitem__(self, n):
        fileid = self._walker[n]
        item = load_vctk_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_txt,
            self._folder_audio,
            self._folder_txt,
        )

        # Legacy
        waveform = item["waveform"]
        if self.transform is not None:
            waveform = self.transform(waveform)
        item["waveform"] = waveform

        # Legacy
        utterance = item["utterance"]
        if self.target_transform is not None:
            utterance = self.target_transform(utterance)
        item["utterance"] = utterance

        if self.return_dict:
            return item

        # Legacy
        return item["waveform"], item["utterance"]

    def __len__(self):
        return len(self._walker)
