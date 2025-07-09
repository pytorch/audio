#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.nn import Module

from .common_utils import _get_feat_lens_paths
from torchaudio.utils import load_torchcodec

_LG = logging.getLogger(__name__)
_DEFAULT_DEVICE = torch.device("cpu")


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
    assert 1 <= rank <= num_rank, f"invalid rank/num_rank {rank}/{num_rank}"
    assert num_lines > 0, f"Found {num_lines} files, make sure you specify the correct root directory"
    start = round(num_lines / num_rank * (rank - 1))
    end = round(num_lines / num_rank * rank)
    _LG.info(f"rank {rank} of {num_rank}, process {end-start} " f"({start}-{end}) out of {num_lines}")
    return start, end


def extract_feature_mfcc(
    path: str,
    device: torch.device,
    sample_rate: int,
) -> Tensor:
    r"""Extract MFCC features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = load_torchcodec(path)
    assert sr == sample_rate
    feature_extractor = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": 400, "hop_length": 160, "center": False}
    ).to(device)
    waveform = waveform[0].to(device)
    mfccs = feature_extractor(waveform)  # (freq, time)
    deltas = torchaudio.functional.compute_deltas(mfccs)
    ddeltas = torchaudio.functional.compute_deltas(deltas)
    concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
    feat = concat.transpose(0, 1)  # (time, freq)
    return feat


def extract_feature_hubert(
    path: str,
    device: torch.device,
    sample_rate: int,
    model: Module,
    layer_index: int,
) -> Tensor:
    r"""Extract HuBERT features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.
        model (Module): The loaded ``HuBERTPretrainModel`` model.
        layer_index (int): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output).

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = load_torchcodec(path)
    assert sr == sample_rate
    waveform = waveform.to(device)
    with torch.inference_mode():
        feat = model.wav2vec2.extract_features(waveform, num_layers=layer_index)[0][-1][0]  # (time, feat_dim)
    return feat


def _load_state(model: Module, checkpoint_path: Path, device=_DEFAULT_DEVICE) -> Module:
    """Load weights from HuBERTPretrainModel checkpoint into hubert_pretrain_base model.
    Args:
        model (Module): The hubert_pretrain_base model.
        checkpoint_path (Path): The model checkpoint.
        device (torch.device, optional): The device of the model. (Default: ``torch.device("cpu")``)

    Returns:
        (Module): The pretrained model.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model


def dump_features(
    tsv_file: Union[str, Path],
    out_dir: Union[str, Path],
    split: str,
    rank: int,
    num_rank: int,
    device: torch.device,
    feature_type: str = "mfcc",
    layer_index: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
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
        layer_index (int or None, optional): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output). Only active when ``feature_type``
            is set to ``hubert``. (Default: ``None``)
        checkpoint_path(Path or None, optional): The checkpoint path of ``torchaudio.models.HuBERTPretrainModel``.
            Only active when ``feature_type`` is set to ``hubert``. (Default: ``None``)
        sample_rate (int, optional): The sample rate of the audio. (Default: ``16000``)

    Returns:
        None
    """
    if feature_type not in ["mfcc", "hubert"]:
        raise ValueError(f"Expected feature type to be 'mfcc' or 'hubert'. Found {feature_type}.")
    if feature_type == "hubert" and layer_index is None:
        assert ValueError("Please set the layer_index for HuBERT feature.")
    features = []
    lens = []
    out_dir = Path(out_dir)

    feat_path, len_path = _get_feat_lens_paths(out_dir, split, rank, num_rank)

    if feature_type == "hubert":
        from torchaudio.models import hubert_pretrain_base

        model = hubert_pretrain_base()
        model.to(device)
        model = _load_state(model, checkpoint_path, device)

    with open(tsv_file, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), num_rank, rank)
        lines = lines[start:end]
        for line in lines:
            path, nsample = line.split("\t")
            path = f"{root}/{path}"
            nsample = int(nsample)
            if feature_type == "mfcc":
                feature = extract_feature_mfcc(path, device, sample_rate)
            else:
                feature = extract_feature_hubert(path, device, sample_rate, model, layer_index)
            features.append(feature.cpu())
            lens.append(feature.shape[0])
    features = torch.cat(features)
    lens = torch.Tensor(lens)
    torch.save(features, feat_path)
    torch.save(lens, len_path)
    _LG.info(f"Finished dumping features for rank {rank} of {num_rank} successfully")
