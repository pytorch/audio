#!/usr/bin/env python3
"""Train the SentencePiece model by using the transcripts of TED-LIUM release 3 training set.
Example:
python train_spm.py --tedlium-path /home/datasets/
"""
import io
import logging
import os
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import sentencepiece as spm

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
        "--output-file",
        default=pathlib.Path("./spm_bpe_500.model"),
        type=pathlib.Path,
        help="File to save model to. (Default: './spm_bpe_500.model')",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _extract_train_text(tedlium_path, output_dir):
    stm_path = tedlium_path / "TEDLIUM_release-3/data/stm/"
    transcripts = []
    for file in sorted(os.listdir(stm_path)):
        if file.endswith(".stm"):
            file = os.path.join(stm_path, file)
            with open(file) as f:
                for line in f.readlines():
                    talk_id, _, speaker_id, start_time, end_time, identifier, transcript = line.split(" ", 6)
                    if transcript == "ignore_time_segment_in_scoring\n":
                        continue
                    else:
                        transcript = transcript.replace("<unk>", "<garbage>").replace("\n", "")
                        transcripts.append(transcript)

    return transcripts


def train_spm(input):
    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(input),
        vocab_size=500,
        model_type="bpe",
        input_sentence_size=-1,
        character_coverage=1.0,
        user_defined_symbols=["<garbage>"],
        bos_id=0,
        pad_id=1,
        eos_id=2,
        unk_id=3,
    )
    return model_writer.getvalue()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    transcripts = _extract_train_text(args.tedlium_path, args.output_dir)
    model = train_spm(transcripts)

    with open(args.output_file, "wb") as f:
        f.write(model)

    logger.info("Successfully trained the sentencepiece model")


if __name__ == "__main__":
    cli_main()
