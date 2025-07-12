#!/usr/bin/env python3
"""Evaluate the lightning module by loading the checkpoint, the SentencePiece model, and the global_stats.json.

Example:
python eval.py --model-type tedlium3 --checkpoint-path ./experiments/checkpoints/epoch=119-step=254999.ckpt
    --dataset-path ./datasets/tedlium --sp-model-path ./spm_bpe_500.model
"""
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import torchaudio
from common import MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_MUSTC, MODEL_TYPE_TEDLIUM3
from librispeech.lightning import LibriSpeechRNNTModule
from mustc.lightning import MuSTCRNNTModule


logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def run_eval_subset(model, dataloader, subset):
    total_edit_distance = 0
    total_length = 0
    with torch.no_grad():
        for idx, (batch, transcripts) in enumerate(dataloader):
            actual = transcripts[0]
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.info(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.info(f"Final WER for {subset} set: {total_edit_distance / total_length}")


def run_eval(model, model_type):
    if model_type == MODEL_TYPE_LIBRISPEECH:
        dataloader = model.test_dataloader()
        run_eval_subset(model, dataloader, "test")
    elif model_type == MODEL_TYPE_MUSTC:
        dev_loader = model.dev_dataloader()
        test_common_loader = model.test_common_dataloader()
        test_he_loader = model.test_he_dataloader()
        run_eval_subset(model, dev_loader, "dev")
        run_eval_subset(model, test_common_loader, "tst-COMMON")
        run_eval_subset(model, test_he_loader, "tst-HE")
    else:
        raise ValueError(f"Encountered unsupported model type {model_type}.")


def get_lightning_module(args):
    if args.model_type == MODEL_TYPE_LIBRISPEECH:
        return LibriSpeechRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            librispeech_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    elif args.model_type == MODEL_TYPE_MUSTC:
        return MuSTCRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            mustc_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--model-type", type=str, choices=[MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3, MODEL_TYPE_MUSTC], required=True
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to dataset.",
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--use-cuda",
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
    run_eval(model, args.model_type)


if __name__ == "__main__":
    cli_main()
