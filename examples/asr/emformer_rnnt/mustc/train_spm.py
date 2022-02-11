"""Train the SentencePiece model by using the transcripts of TED-LIUM release 3 training set.
Example:
python train_spm.py --tedlium-path /home/datasets/
"""

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
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    text_path = args.mustc_path / "en-de/data/train/txt/train.en"

    spm.SentencePieceTrainer.train(
        input=text_path,
        vocab_size=3000,
        model_prefix="spm_bpe_3000",
        model_type="bpe",
        input_sentence_size=100000000,
        character_coverage=1.0,
        bos_id=0,
        pad_id=1,
        eos_id=2,
        unk_id=3,
    )
    logger.info("Successfully trained the sentencepiece model")


if __name__ == "__main__":
    cli_main()
