from pathlib import Path
from typing import Union

import torch
import torchaudio
import yaml


FOLDER_IN_ARCHIVE = "en-de"
SAMPLE_RATE = 16000


class MUSTC(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        language: str = "en",
        subset: str = "train",
    ):
        root = Path(root)
        data_dir = root / folder_in_archive / "data" / subset
        wav_dir = data_dir / "wav"
        yaml_path = data_dir / "txt" / f"{subset}.yaml"
        trans_path = data_dir / "txt" / f"{subset}.{language}"
        with open(yaml_path, "r") as stream:
            file_list = yaml.safe_load(stream)
        with open(trans_path, "r") as f:
            self.trans_list = f.readlines()
        assert len(file_list) == len(self.trans_list)
        self.idx_target_lengths = []
        self.wav_list = []
        for idx, item in enumerate(file_list):
            offset = int(item["offset"] * SAMPLE_RATE)
            duration = int(item["duration"] * SAMPLE_RATE)
            self.idx_target_lengths.append((idx, item["duration"]))
            file_path = wav_dir / item["wav"]
            self.wav_list.append((file_path, offset, duration))

    def _get_mustc_item(self, idx):
        file_path, offset, duration = self.wav_list[idx]
        waveform, sr = torchaudio.load(file_path, frame_offset=offset, num_frames=duration)
        assert sr == SAMPLE_RATE
        transcript = self.trans_list[idx].replace("\n", "")
        return (waveform, transcript)

    def __getitem__(self, idx):
        return self._get_mustc_item(idx)

    def __len__(self):
        return len(self.wav_list)
