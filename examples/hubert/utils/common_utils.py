#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: create tsv files for training (and valiation).
"""

import re
from pathlib import Path
from typing import (
    Union,
)

import torch
import torchaudio


def create_tsv(
    root_dir: Union[str, Path],
    out_dir: Union[str, Path],
    dataset: str = "librispeech",
    valid_percent: float = 0.01,
    extension: str = "flac",
) -> None:
    """Create file lists for training and validation.
    Args:
        root_dir (str or Path): The directory of the dataset.
        out_dir (str or Path): The directory to store the file lists.
        dataset (str, optional): The dataset to use. Options:\
            [``librispeech``, ``libri-light``] (Default: ``librispeech``)
        valid_percent (float, optional): the percentage of data for validation
        extension (str, optional): The extention of audio files.
    """
    assert valid_percent >= 0 and valid_percent <= 1.0

    torch.manual_seed(0)
    root_dir = Path(root_dir)
    out_dir = Path(out_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    valid_f = (
        open(out_dir / f"{dataset}_valid.tsv", "w")
        if valid_percent > 0
        else None
    )
    search_pattern = ".*train.*"
    with open(out_dir / f"{dataset}_train.tsv", "w") as train_f:
        print(root_dir, file=train_f)

        if valid_f is not None:
            print(root_dir, file=valid_f)

        for fname in root_dir.glob(f"**/*.{extension}"):
            if re.match(search_pattern, str(fname)):
                frames = torchaudio.info(fname).num_frames
                dest = train_f if torch.rand(1) > valid_percent else valid_f
                print(
                    f"{fname.relative_to(root_dir)}\t{frames}", file=dest
                )
    if valid_f is not None:
        valid_f.close()
