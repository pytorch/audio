#!/usr/bin/env python3
"""Train the SentencePiece model by using the transcripts of MuST-C release v2.0 training set.
Example:
python train_spm.py --mustc-path /home/datasets/
"""
import io
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import sentencepiece as spm

logger = logging.getLogger(__name__)


def _parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--mustc-path",
        required=True,
        type=pathlib.Path,
        help="Path to MUST-C dataset.",
    )
    parser.add_argument(
        "--output-file",
        default=pathlib.Path("./spm_bpe_500.model"),
        type=pathlib.Path,
        help="File to save model to. (Default: './spm_bpe_500.model')",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def train_spm(input):
    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(input),
        model_writer=model_writer,
        vocab_size=500,
        model_type="bpe",
        input_sentence_size=-1,
        character_coverage=1.0,
        bos_id=0,
        pad_id=1,
        eos_id=2,
        unk_id=3,
    )
    return model_writer.getvalue()


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    with open(args.mustc_path / "en-de/data/train/txt/train.en") as f:
        lines = [line.replace("\n", "") for line in f]
    model = train_spm(lines)

    with open(args.output_file, "wb") as f:
        f.write(model)

    logger.info("Successfully trained the sentencepiece model")


if __name__ == "__main__":
    cli_main()
