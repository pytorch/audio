import os

import torchaudio
import torchvision
from torch.utils.data import Dataset


def _load_list(args, *filenames):
    output = []
    length = []
    for filename in filenames:
        filepath = os.path.join(args.root_dir, "labels", filename)
        for line in open(filepath).read().splitlines():
            dataset, rel_path, input_length = line.split(",")[0], line.split(",")[1], line.split(",")[2]
            path = os.path.normpath(os.path.join(args.root_dir, dataset, rel_path[:-4] + ".mp4"))
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


def load_item(path, modality):
    if modality == "video":
        return (load_video(path), load_transcript(path))
    if modality == "audio":
        return (load_audio(path), load_transcript(path))
    if modality == "audiovisual":
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
            self.files, self.lengths = _load_list(self.args, "lrs3_train_transcript_lengths_seg16s.csv")
        if subset == "val":
            self.files, self.lengths = _load_list(self.args, "lrs3_test_transcript_lengths_seg16s.csv")
        if subset == "test":
            self.files, self.lengths = _load_list(self.args, "lrs3_test_transcript_lengths_seg16s.csv")

    def __getitem__(self, n):
        path = self.files[n]
        return load_item(path, self.args.modality)

    def __len__(self) -> int:
        return len(self.files)
