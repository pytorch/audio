import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import torchaudio
from torchaudio.prototype.pipelines import EMFORMER_RNNT_BASE_TEDLIUM3


logger = logging.getLogger(__name__)


def run_eval_streaming(args):
    dataset = torchaudio.datasets.TEDLIUM(args.tedlium_path, release="release3", subset="test")
    decoder = EMFORMER_RNNT_BASE_TEDLIUM3.get_decoder()
    token_processor = EMFORMER_RNNT_BASE_TEDLIUM3.get_token_processor()
    feature_extractor = EMFORMER_RNNT_BASE_TEDLIUM3.get_feature_extractor()
    streaming_feature_extractor = EMFORMER_RNNT_BASE_TEDLIUM3.get_streaming_feature_extractor()

    hop_length = EMFORMER_RNNT_BASE_TEDLIUM3.hop_length
    num_samples_segment = EMFORMER_RNNT_BASE_TEDLIUM3.segment_length * hop_length
    num_samples_segment_right_context = (
        num_samples_segment + EMFORMER_RNNT_BASE_TEDLIUM3.right_context_length * hop_length
    )

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
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = _parse_args()
    _init_logger(args.debug)
    run_eval_streaming(args)


if __name__ == "__main__":
    cli_main()
