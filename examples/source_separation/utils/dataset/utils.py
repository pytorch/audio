from typing import List
from functools import partial
from collections import namedtuple

import torch

from . import wsj0mix

Batch = namedtuple("Batch", ["mix", "src", "mask"])


def get_dataset(dataset_type, root_dir, num_speakers, sample_rate):
    if dataset_type == "wsj0mix":
        train = wsj0mix.WSJ0Mix(root_dir / "tr", num_speakers, sample_rate)
        validation = wsj0mix.WSJ0Mix(root_dir / "cv", num_speakers, sample_rate)
        evaluation = wsj0mix.WSJ0Mix(root_dir / "tt", num_speakers, sample_rate)
    else:
        raise ValueError(f"Unexpected dataset: {dataset_type}")
    return train, validation, evaluation


def _fix_num_frames(sample: wsj0mix.SampleType, target_num_frames: int, random_start=False):
    """Ensure waveform has exact number of frames by slicing or padding"""
    mix = sample[1]  # [1, num_frames]
    src = torch.cat(sample[2], 0)  # [num_sources, num_frames]

    num_channels, num_frames = src.shape
    if num_frames >= target_num_frames:
        if random_start and num_frames > target_num_frames:
            start_frame = torch.randint(num_frames - target_num_frames, [1])
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
        mix, src, mask = _fix_num_frames(sample, target_num_frames, random_start=True)

        mixes.append(mix)
        srcs.append(src)
        masks.append(mask)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0), torch.stack(masks, 0))


def collate_fn_wsj0mix_test(samples: List[wsj0mix.SampleType]):
    max_num_frames = max(s[1].shape[-1] for s in samples)

    mixes, srcs, masks = [], [], []
    for sample in samples:
        mix, src, mask = _fix_num_frames(sample, max_num_frames, random_start=False)

        mixes.append(mix)
        srcs.append(src)
        masks.append(mask)

    return Batch(torch.stack(mixes, 0), torch.stack(srcs, 0), torch.stack(masks, 0))


def get_collate_fn(dataset_type, mode, sample_rate=None, duration=4):
    assert mode in ["train", "test"]
    if dataset_type == "wsj0mix":
        if mode == 'train':
            if sample_rate is None:
                raise ValueError("sample_rate is not given.")
            return partial(collate_fn_wsj0mix_train, sample_rate=sample_rate, duration=duration)
        return collate_fn_wsj0mix_test
    raise ValueError(f"Unexpected dataset: {dataset_type}")
