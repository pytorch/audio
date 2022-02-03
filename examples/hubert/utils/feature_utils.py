#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
import logging
from pathlib import Path
from typing import (
    Tuple,
    Union,
)

import torch
import torchaudio
from torch import Tensor

from .common_utils import _get_feat_lens_paths

_LG = logging.getLogger(__name__)


def get_shard_range(num_lines: int, num_rank: int, rank: int) -> Tuple[int, int]:
    r"""Get the range of indices for the current rank in multi-processing.
    Args:
        num_lines (int): The number of lines to process.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        rank (int): The rank in the multi-processing.

    Returns:
        (int, int):
        int: The start index for the current rank.
        int: The end index for the current rank.
    """
    assert 0 <= rank < num_rank, f"invalid rank/num_rank {rank}/{num_rank}"
    assert num_lines > 0, f"Found {num_lines} files, make sure you specify the correct root directory"
    start = round(num_lines / num_rank * rank)
    end = round(num_lines / num_rank * (rank + 1))
    _LG.info(f"rank {rank} of {num_rank}, process {end-start} " f"({start}-{end}) out of {num_lines}")
    return start, end


def extract_feature(
    path: str,
    device: torch.device,
    feature_type: str,
    sample_rate: int,
) -> Tensor:
    r"""Extract features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        feature_type (str): The type of the desired feature. Options: [``mfcc``, ``hubert``].
        sample_rate (int): The sample rate of the audio.

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = torchaudio.load(path)
    assert sr == sample_rate
    waveform = waveform[0].to(device)
    if feature_type == "mfcc":
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "center": False}
        ).to(device)
        mfccs = feature_extractor(waveform)  # (freq, time)
        # mfccs = torchaudio.compliance.kaldi.mfcc(
        #     waveform=waveform,
        #     sample_frequency=sample_rate,
        #     use_energy=False,
        # )  # (time, freq)
        # mfccs = mfccs.transpose(0, 1)  # (freq, time)
        deltas = torchaudio.functional.compute_deltas(mfccs)
        ddeltas = torchaudio.functional.compute_deltas(deltas)
        concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
        concat = concat.transpose(0, 1)  # (time, freq)
        return concat


def dump_features(
    tsv_file: Union[str, Path],
    out_dir: Union[str, Path],
    split: str,
    rank: int,
    num_rank: int,
    device: torch.device,
    feature_type: str = "mfcc",
    sample_rate: int = 16_000,
) -> None:
    r"""Dump the feature tensors given a ``.tsv`` file list. The feature and lengths tensors
        will be stored under ``out_dir`` directory.
    Args:
        tsv_file (str or Path): The path of the tsv file.
        out_dir (str or Path): The directory to store the feature tensors.
        split (str): The split of data. Options: [``train``, ``valid``].
        rank (int): The rank in the multi-processing.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        feature_type (str, optional): The type of the desired feature. Options: [``mfcc``, ``hubert``].
            (Default: ``mfcc``)
        sample_rate (int, optional): The sample rate of the audio. (Default: 16000)

    Returns:
        None
    """
    if feature_type not in ["mfcc", "hubert"]:
        raise ValueError("Unexpected feature type.")
    features = []
    lens = []
    out_dir = Path(out_dir)

    feat_path, len_path = _get_feat_lens_paths(out_dir, split, rank, num_rank)
    with open(tsv_file, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), num_rank, rank)
        lines = lines[start:end]
        for line in lines:
            path, nsample = line.split("\t")
            path = f"{root}/{path}"
            nsample = int(nsample)
            feature = extract_feature(path, device, feature_type, sample_rate)
            features.append(feature.cpu())
            lens.append(feature.shape[0])
    features = torch.cat(features)
    lens = torch.Tensor(lens)
    torch.save(features, feat_path)
    torch.save(lens, len_path)
    _LG.info(f"Finished dumping features for rank {rank} of {num_rank} successfully")
