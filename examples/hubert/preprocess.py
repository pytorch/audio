#!/usr/bin/env python3
"""This is the preprocessing script for HuBERT model training.
The script includes:
    - File list creation
    - MFCC/HuBERT feature extraction
    - KMeans clustering model training
    - Pseudo-label generation
"""
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from multiprocessing import Pool
from pathlib import Path

import torch
from utils import (
    create_tsv,
    dump_features,
    learn_kmeans,
    get_km_label,
)


def _init_logger(debug=False):
    message_fmt = "%(levelname)5s: %(funcName)10s: %(message)s" if debug else "%(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"%(asctime)s: {message_fmt}",
    )


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug log")
    parser.add_argument("--dataset", default="librispeech", type=str, choices=["librispeech", "librilight"])
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="The path to the directory where the directory ``LibriSpeech`` or ``LibriLight`` is stored.",
    )
    parser.add_argument("--num-rank", default=5, type=int)
    parser.add_argument("--feat-type", default="mfcc", type=str)
    parser.add_argument("--use-gpu", default=False, type=bool)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        help="The directory to store the experiment outputs.",
    )
    parser.add_argument(
        "--num-cluster",
        default=100,
        type=int,
        help="The number of clusters for KMeans clustering.",
    )
    args = parser.parse_args()
    return args


def main(args):
    _init_logger(args.debug)

    if not args.exp_dir.exists():
        args.exp_dir.mkdir()
    tsv_dir = args.exp_dir / "tsv"
    feat_dir = args.exp_dir / args.feat_type
    km_dir = args.exp_dir / "km_model"
    label_dir = args.exp_dir / "label"

    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create file lists for training and validation (optional)
    create_tsv(args.root_dir, tsv_dir)

    # Extract features for KMeans clustering
    if not feat_dir.exists():
        feat_dir.mkdir()

    for split in ["train", "valid"]:
        p = Pool(args.num_rank)
        inputs = [
            (
                tsv_dir / f"{args.dataset}_{split}.tsv",
                feat_dir,
                split,
                rank,
                args.num_rank,
                device,
                args.feat_type,
                16_000,
            )
            for rank in range(args.num_rank)
        ]
        _ = p.starmap(dump_features, inputs)
        p.close()
        p.join()

    # Fit KMeans clustering model
    learn_kmeans(
        feat_dir,
        "train",
        args.num_rank,
        km_dir,
        args.num_cluster,
    )

    # Predict labels for MFCC features
    for split in ["train", "valid"]:
        get_km_label(
            feat_dir,
            km_dir,
            label_dir,
            split,
            args.num_rank,
            device,
        )


if __name__ == "__main__":
    main(_parse_args())
