import logging
import pathlib
from argparse import ArgumentParser

import torch
import torchaudio
from common import MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH
from torchaudio.prototype.pipelines import EMFORMER_RNNT_BASE_TEDLIUM3


logger = logging.getLogger()


def get_pipeline_bundle(args):
    if args.model_type == MODEL_TYPE_LIBRISPEECH:
        dataset = torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="test-clean")
        decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder()
        token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()
        feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
        streaming_feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_streaming_feature_extractor()
        hop_length = EMFORMER_RNNT_BASE_LIBRISPEECH.hop_length
        num_samples_segment = EMFORMER_RNNT_BASE_LIBRISPEECH.segment_length * hop_length
        num_samples_segment_right_context = (
            num_samples_segment + EMFORMER_RNNT_BASE_LIBRISPEECH.right_context_length * hop_length
        )
    elif args.model_type == MODEL_TYPE_TEDLIUM3:
        dataset = torchaudio.datasets.TEDLIUM(args.dataset_path, release="release3", subset="test")
        decoder = EMFORMER_RNNT_BASE_TEDLIUM3.get_decoder()
        token_processor = EMFORMER_RNNT_BASE_TEDLIUM3.get_token_processor()
        feature_extractor = EMFORMER_RNNT_BASE_TEDLIUM3.get_feature_extractor()
        streaming_feature_extractor = EMFORMER_RNNT_BASE_TEDLIUM3.get_streaming_feature_extractor()
        hop_length = EMFORMER_RNNT_BASE_TEDLIUM3.hop_length
        num_samples_segment = EMFORMER_RNNT_BASE_TEDLIUM3.segment_length * hop_length
        num_samples_segment_right_context = (
            num_samples_segment + EMFORMER_RNNT_BASE_TEDLIUM3.right_context_length * hop_length
        )
    else:
        raise ValueError(f"Encountered unsupported model type {args.model_type}.")

    return (
        dataset,
        decoder,
        token_processor,
        feature_extractor,
        streaming_feature_extractor,
        hop_length,
        num_samples_segment,
        num_samples_segment_right_context,
    )


def run_eval_streaming(args):
    (
        dataset,
        decoder,
        token_processor,
        feature_extractor,
        streaming_feature_extractor,
        hop_length,
        num_samples_segment,
        num_samples_segment_right_context,
    ) = get_pipeline_bundle(args)

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
            hypothesis = hypos[0]
            transcript = token_processor(hypothesis.tokens, lstrip=False)
            print(transcript, end="", flush=True)
        print()

        # Non-streaming decode.
        with torch.no_grad():
            features, length = feature_extractor(waveform)
            hypos = decoder(features, length, 10)
        print(token_processor(hypos[0].tokens))
        print()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=[MODEL_TYPE_LIBRISPEECH, MODEL_TYPE_TEDLIUM3], required=True)
    parser.add_argument(
        "--dataset_path",
        type=pathlib.Path,
        help="Path to dataset.",
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
