import logging
import math
import pathlib
from argparse import ArgumentParser

import sentencepiece as spm

import torch
import torchaudio
from lightning import ConformerCTCModule
from transforms import get_data_module


logger = logging.getLogger()


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def run_eval(args):
    sp_model = spm.SentencePieceProcessor(model_file=str(args.sp_model_path))

    # https://pytorch.org/audio/main/generated/torchaudio.models.decoder.ctc_decoder.html
    inference_args = {
        "nbest": 3,
        "beam_size": 50,
        "beam_size_token": None,
        "beam_threshold": 50,
        "lm_weight": 2,
        "word_score": 0,
        "unk_score": -math.inf,
        "sil_score": 0,
    }

    model = ConformerCTCModule.load_from_checkpoint(
        args.checkpoint_path, sp_model=sp_model, inference_args=inference_args
    ).eval()
    data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), str(args.sp_model_path))

    if args.use_cuda:
        model = model.to(device="cuda")

    total_edit_distance = 0
    total_length = 0
    dataloader = data_module.test_dataloader()
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            actual = sample[0][2]
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            print(f"[{idx}][predicted]\t{predicted}")
            print(f"[{idx}][actual   ]\t{actual}")
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.warning(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.warning(f"Final WER: {total_edit_distance / total_length}")


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    cli_main()
