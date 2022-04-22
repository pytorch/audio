#!/usr/bin/env python3
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import torchaudio
from torchaudio.prototype.pipelines import EMFORMER_RNNT_BASE_TEDLIUM3


logger = logging.getLogger(__name__)


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def _eval_subset(tedlium_path, subset, feature_extractor, decoder, token_processor, use_cuda):
    total_edit_distance = 0
    total_length = 0
    if subset == "dev":
        dataset = torchaudio.datasets.TEDLIUM(tedlium_path, release="release3", subset="dev")
    elif subset == "test":
        dataset = torchaudio.datasets.TEDLIUM(tedlium_path, release="release3", subset="test")
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            waveform = sample[0].squeeze()
            if use_cuda:
                waveform = waveform.to(device="cuda")
            actual = sample[2].replace("\n", "")
            if actual == "ignore_time_segment_in_scoring":
                continue
            features, length = feature_extractor(waveform)
            hypos = decoder(features, length, 20)
            hypothesis = hypos[0]
            hypothesis = token_processor(hypothesis[0])
            total_edit_distance += compute_word_level_distance(actual, hypothesis)
            total_length += len(actual.split())
            if idx % 100 == 0:
                print(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    print(f"Final WER for {subset} set: {total_edit_distance / total_length}")


def run_eval_pipeline(args):
    decoder = EMFORMER_RNNT_BASE_TEDLIUM3.get_decoder()
    token_processor = EMFORMER_RNNT_BASE_TEDLIUM3.get_token_processor()
    feature_extractor = EMFORMER_RNNT_BASE_TEDLIUM3.get_feature_extractor()

    if args.use_cuda:
        feature_extractor = feature_extractor.to(device="cuda").eval()
        decoder = decoder.to(device="cuda")
    _eval_subset(args.tedlium_path, "dev", feature_extractor, decoder, token_processor, args.use_cuda)
    _eval_subset(args.tedlium_path, "test", feature_extractor, decoder, token_processor, args.use_cuda)


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--tedlium-path",
        type=pathlib.Path,
        help="Path to TED-LIUM release 3 dataset.",
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
    run_eval_pipeline(args)


if __name__ == "__main__":
    cli_main()
