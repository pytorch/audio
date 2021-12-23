#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
import logging
from pathlib import Path
from typing import (
    Tuple,
)

import joblib
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor

from .common_utils import _get_feat_lens_paths, _get_model_path

_LG = logging.getLogger(__name__)


def load_feature(
    feat_dir: Path,
    split: str,
    num_rank: int,
) -> Tuple[Tensor, Tensor]:
    r"""Loading features from pre-saved `.pt` files.
    Args:
        feat_dir (Path): The directory that stores the feature files.
        split (str): The split of data. Options: [``train``, ``valid``].
        num_rank (int): The number of ranks for multi-processing in feature extraction.

    Returns:
        (Tensor, Tensor)
        Tensor: The concatenated feature tensor of shape `(frame, feature_dim)`.
        Tensor: The lengths tensor of shape `(num_utterance,)`.
    """
    feats = []
    lens = []
    for rank in range(num_rank):
        feat_path, len_path = _get_feat_lens_paths(feat_dir, split, rank, num_rank)
        feat = torch.load(feat_path)
        length = torch.load(len_path)
        feats.append(feat)
        lens.append(length)
    feats = torch.cat(feats)
    lens = torch.cat(lens)
    return feats, lens


def learn_kmeans(
    feat_dir: Path,
    split: str,
    num_rank: int,
    km_dir: Path,
    n_clusters: int,
    init: str = "k-means++",
    max_iter: int = 100,
    batch_size: int = 10000,
    tol: float = 0.0,
    n_init: int = 20,
    reassignment_ratio: float = 0.0,
    max_no_improvement: int = 100,
) -> None:
    r"""Build and train the KMeans clustering model. The model is saved in "{km_dir}/model.pt"
    Args:
        feat_dir (Path): The directory that stores the feature files.
        split (str): The split of data. Options: [``train``, ``valid``].
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        km_dir (Path): The directory to store the KMeans clustering model.
        n_clusters (int): The number of clusters.
        init (str, optional): Method for initialization. Options: [``k-means++``, ``random``].
            (Default: ``k-means++``)
        max_iter (int, optional): Maximum number of iterations over the complete dataset. (Default: 100)
        batch_size (int, optional): Batch size for training the KMeans clustering model. (Default: 10000)
        tol (float, optional): Control early stopping based on the relative center changes as measured by a smoothed,
            variance-normalized of the mean center squared position changes. (Default: 0.0)
        n_init (int, optional): Number of random initializations that are tried. (Default: 20)
        reassignment_ratio (float, optional): Control the fraction of the maximum number of counts for a center
            to be reassigned. A higher value means that low count centers are more easily reassigned. (Default: 0.0)
        max_no_improvement (int, optional): Control early stopping based on the consecutive number of mini batches
            that does not yield an improvement on the smoothed inertia. (Default: 100)

    Returns:
        None
    """
    if not km_dir.exists():
        km_dir.mkdir()

    km_model = MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=0,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

    feats, _ = load_feature(
        feat_dir,
        split,
        num_rank,
    )
    feats = feats.numpy()
    km_model.fit(feats)
    km_path = _get_model_path(km_dir)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feats) / len(feats)
    _LG.info("Total intertia: %.5f", inertia)
    _LG.info("Finished training the KMeans clustering model successfully")


class ApplyKmeans(object):
    def __init__(self, km_path, device):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist.argmin(dim=1).cpu().numpy()


def get_km_label(
    feat_dir: Path,
    km_dir: Path,
    label_dir: Path,
    split: str,
    num_rank: int,
    device: torch.device,
) -> None:
    r"""Predict the labels by the KMeans clustering model.
    Args:
        feat_dir (Path): The directory that stores the dumped features.
        km_dir (Path): The directory that stores the KMeans model.
        label_dir (Path): The directory to save the predicted labels.
        split (str): The split of data. Options: [``train``, ``valid``].
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
    Returns:
        None
    """
    if not label_dir.exists():
        label_dir.mkdir()

    km_path = _get_model_path(km_dir)
    label_path = label_dir / f"label_{split}.pt"
    apply_kmeans = ApplyKmeans(km_path, device)
    feats, lens = load_feature(
        feat_dir,
        split,
        num_rank,
    )
    feats = feats
    lens = lens.long()
    offset = 0
    assert feats.shape[0] == lens.sum()
    with open(label_path, "w") as f:
        for i in range(lens.shape[0]):
            feat = feats[offset : offset + lens[i]].to(device)
            offset += lens[i]
            label = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, label)) + "\n")
    _LG.info("Finished predicting labels successfully")
