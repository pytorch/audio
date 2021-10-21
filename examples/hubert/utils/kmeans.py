from pathlib import Path
from typing import (
    Tuple,
)
import logging
from sklearn.cluster import MiniBatchKMeans
from torch import Tensor
import joblib
import torch


logger = logging.getLogger("learn_kmeans")


def load_feature(
    feat_dir: Path,
    split: str,
    num_rank: int,
) -> Tuple[Tensor, Tensor]:
    feats = []
    lens = []
    for rank in range(num_rank):
        feat_path = feat_dir / f"{split}_{rank}_{num_rank}.pt"
        len_path = feat_dir / f"len_{split}_{rank}_{num_rank}.pt"
        feat = torch.load(feat_path)
        length = torch.load(len_path)
        feats.append(feat)
        lens.append(length)
    feats = torch.cat(feats)
    lens = torch.cat(lens)
    return feats, lens


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def learn_kmeans(
    feat_dir: Path,
    split: str,
    num_rank: int,
    km_dir: Path,
    n_clusters: int,
    seed: int = 0,
    init: str = "k-means++",
    max_iter: int = 100,
    batch_size: int = 10000,
    tol: float = 0.0,
    n_init: int = 20,
    reassignment_ratio: float = 0.0,
    max_no_improvement: int = 100,
) -> None:
    if not km_dir.exists():
        km_dir.mkdir()

    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    feats, _ = load_feature(
        feat_dir,
        split,
        num_rank,
    )
    feats = feats.numpy()
    km_model.fit(feats)
    joblib.dump(km_model, km_dir / "model.pt")

    inertia = -km_model.score(feats) / len(feats)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


class ApplyKmeans(object):
    def __init__(self, km_path, device):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.C)
            + self.Cnorm
        )
        return dist.argmin(dim=1).cpu().numpy()


def get_km_label(
    feat_dir: Path,
    km_dir: Path,
    label_dir: Path,
    split: str,
    num_rank: int,
    device: torch.device,
) -> None:
    if not label_dir.exists():
        label_dir.mkdir()

    km_path = km_dir / "model.pt"
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
            feat = feats[offset:offset + lens[i]].to(device)
            offset += lens[i]
            label = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, label)) + "\n")
    logger.info("finished successfully")
