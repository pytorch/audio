#!/usr/bin/env python3
"""Generate feature statistics for training set.

Example:
python global_stats.py --model-type librispeech --dataset-path /home/librispeech
"""

import json
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import torchaudio
from common import (
    MODEL_TYPE_LIBRISPEECH,
    MODEL_TYPE_MUSTC,
    MODEL_TYPE_TEDLIUM3,
    piecewise_linear_log,
    spectrogram_transform,
)
from mustc.dataset import MUSTC


logger = logging.getLogger()


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--model-type", type=str, choices=[MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3, MODEL_TYPE_MUSTC], required=True
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=pathlib.Path,
        help="Path to dataset. "
        "For LibriSpeech, all of 'train-clean-360', 'train-clean-100', and 'train-other-500' must exist.",
    )
    parser.add_argument(
        "--output-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="File to save feature statistics to. (Default: './global_stats.json')",
    )
    return parser.parse_args()


def generate_statistics(samples):
    E_x = 0
    E_x_2 = 0
    N = 0

    for idx, sample in enumerate(samples):
        mel_spec = spectrogram_transform(sample[0].squeeze()).transpose(1, 0)
        scaled_mel_spec = piecewise_linear_log(mel_spec)
        sum = scaled_mel_spec.sum(0)
        sq_sum = scaled_mel_spec.pow(2).sum(0)
        M = scaled_mel_spec.size(0)

        E_x = E_x * (N / (N + M)) + sum / (N + M)
        E_x_2 = E_x_2 * (N / (N + M)) + sq_sum / (N + M)
        N += M

        if idx % 100 == 0:
            logger.info(f"Processed {idx}")

    return E_x, (E_x_2 - E_x**2) ** 0.5


def get_dataset(args):
    if args.model_type == MODEL_TYPE_LIBRISPEECH:
        return torch.utils.data.ConcatDataset(
            [
                torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-clean-360"),
                torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-clean-100"),
                torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-other-500"),
            ]
        )
    elif args.model_type == MODEL_TYPE_TEDLIUM3:
        return torchaudio.datasets.TEDLIUM(args.dataset_path, release="release3", subset="train")
    elif args.model_type == MODEL_TYPE_MUSTC:
        return MUSTC(args.dataset_path, subset="train")
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")


def cli_main():
    args = parse_args()
    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)
    mean, stddev = generate_statistics(iter(dataloader))

    json_str = json.dumps({"mean": mean.tolist(), "invstddev": (1 / stddev).tolist()}, indent=2)

    with open(args.output_path, "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    cli_main()
