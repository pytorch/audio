from collections import namedtuple
from functools import partial
from typing import List

import torch
from torchaudio.datasets import LibriMix

from . import wsj0mix

Batch = namedtuple("Batch", ["mix", "src", "mask"])


def get_dataset(dataset_type, root_dir, num_speakers, sample_rate, task=None, librimix_tr_split=None):
    if dataset_type == "wsj0mix":
        train = wsj0mix.WSJ0Mix(root_dir / "tr", num_speakers, sample_rate)
        validation = wsj0mix.WSJ0Mix(root_dir / "cv", num_speakers, sample_rate)
        evaluation = wsj0mix.WSJ0Mix(root_dir / "tt", num_speakers, sample_rate)
    elif dataset_type == "librimix":
        train = LibriMix(root_dir, librimix_tr_split, num_speakers, sample_rate, task)
        validation = LibriMix(root_dir, "dev", num_speakers, sample_rate, task)
        evaluation = LibriMix(root_dir, "test", num_speakers, sample_rate, task)
    else:
        raise ValueError(f"Unexpected dataset: {dataset_type}")
    return train, validation, evaluation


def _fix_num_frames(sample: wsj0mix.SampleType, target_num_frames: int, sample_rate: int, random_start=False):
    """Ensure waveform has exact number of frames by slicing or padding"""
    mix = sample[1]  # [1, time]
    src = torch.cat(sample[2], 0)  # [num_sources, time]

    num_channels, num_frames = src.shape
    num_seconds = torch.div(num_frames, sample_rate, rounding_mode="floor")
    target_seconds = torch.div(target_num_frames, sample_rate, rounding_mode="floor")
    if num_frames >= target_num_frames:
        if random_start and num_frames > target_num_frames:
            start_frame = torch.randint(num_seconds - target_seconds + 1, [1]) * sample_rate
            mix = mix[:, start_frame:]
            src = src[:, start_frame:]
        mix = mix[:, :target_num_frames]
        src = src[:, :target_num_frames]
        mask = torch.ones_like(mix)
    else:
        num_padding = target_num_frames - num_frames
        pad = torch.zeros([1, num_padding], dtype=mix.dtype, device=mix.device)
        mix = torch.cat([mix, pad], 1)
        src = torch.cat([src, pad.expand(num_channels, -1)], 1)
        mask = torch.ones_like(mix)
        mask[..., num_frames:] = 0
    return mix, src, mask


def collate_fn_wsj0mix_train(samples: List[wsj0mix.SampleType], sample_rate, duration):
    target_num_frames = int(duration * sample_rate)

    mixes, srcs, masks = [], [], []
    for sample in samples:
        mix, src, mask = _fix_num_frames(sample, target_num_frames, sample_rate, random_start=True)

        mixes.append(mix)
        srcs.append(src)
        masks.append(mask)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0), torch.stack(masks, 0))


def collate_fn_wsj0mix_test(samples: List[wsj0mix.SampleType], sample_rate):
    max_num_frames = max(s[1].shape[-1] for s in samples)

    mixes, srcs, masks = [], [], []
    for sample in samples:
        mix, src, mask = _fix_num_frames(sample, max_num_frames, sample_rate, random_start=False)

        mixes.append(mix)
        srcs.append(src)
        masks.append(mask)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0), torch.stack(masks, 0))


def get_collate_fn(dataset_type, mode, sample_rate=None, duration=4):
    assert mode in ["train", "test"]
    if dataset_type in ["wsj0mix", "librimix"]:
        if mode == "train":
            if sample_rate is None:
                raise ValueError("sample_rate is not given.")
            return partial(collate_fn_wsj0mix_train, sample_rate=sample_rate, duration=duration)
        return partial(collate_fn_wsj0mix_test, sample_rate=sample_rate)
    raise ValueError(f"Unexpected dataset: {dataset_type}")
