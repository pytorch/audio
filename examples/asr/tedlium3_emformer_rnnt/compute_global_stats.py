"""Generate feature statistics for TED-LIUM release 3 training set.
Example:
python compute_global_stats.py --tedlium-path /home/datasets/
"""

import json
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torchaudio
from utils import GAIN, piecewise_linear_log, spectrogram_transform

logger = logging.getLogger(__name__)


def _parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--tedlium-path",
        required=True,
        type=pathlib.Path,
        help="Path to TED-LIUM release 3 dataset.",
    )
    parser.add_argument(
        "--output-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="File to save feature statistics to. (Default: './global_stats.json')",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _compute_stats(dataset):
    E_x = 0.0
    E_x_2 = 0.0
    N = 0.0
    for idx, data in enumerate(dataset):
        waveform = data[0].squeeze()
        mel_spec = spectrogram_transform(waveform)
        scaled_mel_spec = piecewise_linear_log(mel_spec * GAIN)
        mel_sum = scaled_mel_spec.sum(-1)
        mel_sum_sq = scaled_mel_spec.pow(2).sum(-1)
        M = scaled_mel_spec.size(1)

        E_x = E_x * (N / (N + M)) + mel_sum / (N + M)
        E_x_2 = E_x_2 * (N / (N + M)) + mel_sum_sq / (N + M)
        N += M

        if idx % 100 == 0:
            logger.info(f"Processed {idx}")

    return E_x, (E_x_2 - E_x ** 2) ** 0.5


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    dataset = torchaudio.datasets.TEDLIUM(args.tedlium_path, release="release3", subset="train")
    mean, std = _compute_stats(dataset)
    invstd = 1 / std

    stats_dict = {
        "mean": mean.tolist(),
        "invstddev": invstd.tolist(),
    }

    with open(args.output_path, "w") as f:
        json.dump(stats_dict, f, indent=2)


if __name__ == "__main__":
    cli_main()
