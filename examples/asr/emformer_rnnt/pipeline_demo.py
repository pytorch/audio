#!/usr/bin/env python3
"""The demo script for testing the pre-trained Emformer RNNT pipelines.

Example:
python pipeline_demo.py --model-type librispeech --dataset-path ./datasets/librispeech
"""
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torchaudio
from common import MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_MUSTC, MODEL_TYPE_TEDLIUM3
from mustc.dataset import MUSTC
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH, RNNTBundle

logger = logging.getLogger(__name__)


@dataclass
class Config:
    dataset: Callable
    bundle: RNNTBundle


_CONFIGS = {
    MODEL_TYPE_LIBRISPEECH: Config(
        partial(torchaudio.datasets.LIBRISPEECH, url="test-clean"),
        EMFORMER_RNNT_BASE_LIBRISPEECH,
    )
}


def run_eval_streaming(args):
    dataset = _CONFIGS[args.model_type].dataset(args.dataset_path)
    bundle = _CONFIGS[args.model_type].bundle
    decoder = bundle.get_decoder()
    token_processor = bundle.get_token_processor()
    feature_extractor = bundle.get_feature_extractor()
    streaming_feature_extractor = bundle.get_streaming_feature_extractor()
    hop_length = bundle.hop_length
    num_samples_segment = bundle.segment_length * hop_length
    num_samples_segment_right_context = num_samples_segment + bundle.right_context_length * hop_length

    for idx in range(10):
        sample = dataset[idx]
        waveform = sample[0].squeeze()
        # Streaming decode.
        state, hypothesis = None, None
        for idx in range(0, len(waveform), num_samples_segment):
            segment = waveform[idx : idx + num_samples_segment_right_context]
            segment = torch.nn.functional.pad(segment, (0, num_samples_segment_right_context - len(segment)))
            with torch.no_grad():
                features, length = streaming_feature_extractor(segment)
                hypos, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
            hypothesis = hypos
            transcript = token_processor(hypos[0][0], lstrip=True)
            print(transcript, end="\r", flush=True)
        print()

        # Non-streaming decode.
        with torch.no_grad():
            features, length = feature_extractor(waveform)
            hypos = decoder(features, length, 10)
        print(token_processor(hypos[0][0]))
        print()


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model-type", type=str, choices=_CONFIGS.keys(), required=True)
    parser.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        help="Path to dataset.",
        required=True,
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
    run_eval_streaming(args)


if __name__ == "__main__":
    cli_main()
