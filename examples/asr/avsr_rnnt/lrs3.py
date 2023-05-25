import os
from pathlib import Path
from typing import Tuple, Union

import torch
import torchaudio
import torchvision
from torch import Tensor
from torch.utils.data import Dataset


def _load_list(args, *filenames):
    output = []
    length = []
    for filename in filenames:
        filepath = os.path.join(os.path.dirname(args.dataset_path), filename)
        for line in open(filepath).read().splitlines():
            dataset_name, rel_path, input_length = line.split(",")[0], line.split(",")[1], line.split(",")[2]
            path = os.path.normpath(os.path.join(args.dataset_path, rel_path[:-4] + ".mp4"))
            length.append(int(input_length))
            output.append(path)
    return output, length


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path, normalize=True)
    return waveform.transpose(1, 0)


def load_transcript(path):
    transcript_path = path.replace("video_seg", "text_seg")[:-4] + ".txt"
    return open(transcript_path).read().splitlines()[0]


def load_item(path, md):
    if md == "v":
        return (load_video(path), load_transcript(path))
    if md == "a":
        return (load_audio(path), load_transcript(path))
    if md == "av":
        return (load_audio(path), load_video(path), load_transcript(path))


class LRS3(Dataset):
    def __init__(
        self,
        args,
        subset: str = "train",
    ) -> None:

        if subset is not None and subset not in ["train", "val", "test"]:
            raise ValueError("When `subset` is not None, it must be one of ['train', 'val', 'test'].")

        self.args = args

        if subset == "train":
            self._filelist, self._lengthlist = _load_list(self.args, "train_transcript_lengths_seg16s.csv")
        if subset == "val":
            self._filelist, self._lengthlist = _load_list(self.args, "test_transcript_lengths_seg16s.csv")
        if subset == "test":
            self._filelist, self._lengthlist = _load_list(self.args, "test_transcript_lengths_seg16s.csv")

    def __getitem__(self, n):
        path = self._filelist[n]
        return load_item(path, self.args.md)

    def __len__(self) -> int:
        return len(self._filelist)
