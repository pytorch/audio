import logging
import pathlib
from argparse import ArgumentParser

import torch
import torchaudio
from common import MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3
from librispeech.lightning import LibriSpeechRNNTModule
from tedlium3.lightning import TEDLIUM3RNNTModule


logger = logging.getLogger()


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def run_eval(model):
    total_edit_distance = 0
    total_length = 0
    dataloader = model.test_dataloader()
    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(dataloader):
            actual = transcripts[0]
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.info(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.info(f"Final WER: {total_edit_distance / total_length}")


def get_lightning_module(args):
    if args.model_type == MODEL_TYPE_LIBRISPEECH:
        return LibriSpeechRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            librispeech_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.model_type == MODEL_TYPE_TEDLIUM3:
        return TEDLIUM3RNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            tedlium_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=[MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3], required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--global_stats_path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--dataset_path",
        type=pathlib.Path,
        help="Path to dataset.",
    )
    parser.add_argument(
        "--sp_model_path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    if args.use_cuda:
        model = model.to(device="cuda")
    run_eval(model)


if __name__ == "__main__":
    cli_main()
