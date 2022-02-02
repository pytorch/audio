import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import torchaudio
from lightning import RNNTModule


logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def _eval_subset(model, subset):
    total_edit_distance = 0.0
    total_length = 0.0
    if subset == "dev":
        dataloader = model.dev_dataloader()
    else:
        dataloader = model.test_dataloader()
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            actual = sample[0][2].replace("\n", "")
            if actual == "ignore_time_segment_in_scoring":
                continue
            predicted = model(batch)
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.info(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.info(f"Final WER for {subset} set: {total_edit_distance / total_length}")


def run_eval(args):
    model = RNNTModule.load_from_checkpoint(
        args.checkpoint_path,
        tedlium_path=str(args.tedlium_path),
        sp_model_path=str(args.sp_model_path),
        global_stats_path=str(args.global_stats_path),
        reduction="mean",
    ).eval()

    if args.use_cuda:
        model = model.to(device="cuda")
    _eval_subset(model, "dev")
    _eval_subset(model, "test")


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
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
        "--tedlium-path",
        type=pathlib.Path,
        help="Path to TED-LIUM release 3 dataset.",
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


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    run_eval(args)


if __name__ == "__main__":
    cli_main()
