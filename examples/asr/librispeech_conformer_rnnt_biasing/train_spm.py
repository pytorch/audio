#!/usr/bin/env python3
"""Trains a SentencePiece model on transcripts across LibriSpeech train-clean-100, train-clean-360, and train-other-500.
Using unigram wordpiece model and suffix-based wordpieces

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


def train_spm(input, suffix=False):
    model_writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(input),
        model_writer=model_writer,
        vocab_size=600,
        model_type="unigram",
        input_sentence_size=-1,
        character_coverage=1.0,
        treat_whitespace_as_suffix=suffix,
        bos_id=0,
        pad_id=1,
        eos_id=2,
        unk_id=3,
    )
    return model_writer.getvalue()


def parse_args():
    default_output_path = "./spm_unigram_600_100suffix.model"
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--librispeech-path",
        required=True,
        type=pathlib.Path,
        help="Path to LibriSpeech dataset.",
    )
    parser.add_argument(
        "--output-file",
        default=pathlib.Path(default_output_path),
        type=pathlib.Path,
        help=f"File to save model to. (Default: '{default_output_path}')",
    )
    parser.add_argument(
        "--suffix",
        action='store_true',
        help="whether to use suffix-based wordpieces",
    )
    return parser.parse_args()


def run_cli():
    args = parse_args()

    root = args.librispeech_path / "LibriSpeech"
    # Uncomment this for running bpe on full 960-hour data
    # splits = ["train-clean-100", "train-clean-360", "train-other-500"]
    splits = ["train-clean-100"]
    merged_transcripts = []
    for split in splits:
        path = pathlib.Path(root) / split
        merged_transcripts += get_transcripts(path)

    model = train_spm(merged_transcripts, suffix=args.suffix)

    with open(args.output_file, "wb") as f:
        f.write(model)


if __name__ == "__main__":
    run_cli()
