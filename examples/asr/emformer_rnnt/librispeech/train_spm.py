#!/usr/bin/env python3
"""Trains a SentencePiece model on transcripts across LibriSpeech train-clean-100, train-clean-360, and train-other-500.

Example:
python train_spm.py --librispeech-path ./datasets
"""

import io
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import sentencepiece as spm


def get_transcript_text(transcript_path):
    with open(transcript_path) as f:
        return [line.strip().split(" ", 1)[1].lower() for line in f]


def get_transcripts(dataset_path):
    transcript_paths = dataset_path.glob("*/*/*.trans.txt")
    merged_transcripts = []
    for path in transcript_paths:
        merged_transcripts += get_transcript_text(path)
    return merged_transcripts


def train_spm(input):
    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(input),
        model_writer=model_writer,
        vocab_size=4096,
        model_type="bpe",
        input_sentence_size=-1,
        character_coverage=1.0,
        bos_id=0,
        pad_id=1,
        eos_id=2,
        unk_id=3,
    )
    return model_writer.getvalue()


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--librispeech-path",
        required=True,
        type=pathlib.Path,
        help="Path to LibriSpeech dataset.",
    )
    parser.add_argument(
        "--output-file",
        default=pathlib.Path("./spm_bpe_4096.model"),
        type=pathlib.Path,
        help="File to save model to. (Default: './spm_bpe_4096.model')",
    )
    return parser.parse_args()


def run_cli():
    args = parse_args()

    root = args.librispeech_path / "LibriSpeech"
    splits = ["train-clean-100", "train-clean-360", "train-other-500"]
    merged_transcripts = []
    for split in splits:
        path = pathlib.Path(root) / split
        merged_transcripts += get_transcripts(path)

    model = train_spm(merged_transcripts)

    with open(args.output_file, "wb") as f:
        f.write(model)


if __name__ == "__main__":
    run_cli()
