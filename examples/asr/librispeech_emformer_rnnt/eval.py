from argparse import ArgumentParser
import logging
import pathlib

import torch
import torchaudio

from lightning import RNNTModule


logger = logging.getLogger()


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(
        seq1.lower().split(), seq2.lower().split()
    )


def run_eval(args):
    model = RNNTModule.load_from_checkpoint(
        args.checkpoint_path,
        librispeech_path=args.librispeech_path,
        sp_model_path=args.sp_model_path,
        tgt_dict_path=args.tgt_dict_path,
        global_stats_path=args.global_stats_path,
    ).eval()

    if args.use_cuda:
        model = model.to(device="cuda")

    total_edit_distance = 0
    total_length = 0
    dataloader = model.test_dataloader()
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            actual = sample[0][2]
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.info(
                    f"Processed elem {idx}; WER: {total_edit_distance / total_length}"
                )
    logger.info(f"Final WER: {total_edit_distance / total_length}")


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--global_stats_path",
        type=str,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech_path", type=str, help="Path to LibriSpeech datasets.",
    )
    parser.add_argument(
        "--sp_model_path", type=str, help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--tgt_dict_path", type=str, help="Path to fairseq token dictionary.",
    )
    parser.add_argument(
        "--use_cuda", action="store_true", default=False, help="Run using CUDA.",
    )
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    cli_main()
