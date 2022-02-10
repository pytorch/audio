import pathlib
from argparse import ArgumentParser

import torch
import torchaudio
from torchaudio.pipelines import EMFORMER_RNNT_BASE_LIBRISPEECH


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--librispeech_path",
        type=pathlib.Path,
        required=True,
        help="Path to LibriSpeech datasets.",
    )
    args = parser.parse_args()

    dataset = torchaudio.datasets.LIBRISPEECH(args.librispeech_path, url="test-clean")
    decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder()
    token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()
    feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
    streaming_feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_streaming_feature_extractor()

    hop_length = EMFORMER_RNNT_BASE_LIBRISPEECH.hop_length
    num_samples_segment = EMFORMER_RNNT_BASE_LIBRISPEECH.segment_length * hop_length
    num_samples_segment_right_context = (
        num_samples_segment + EMFORMER_RNNT_BASE_LIBRISPEECH.right_context_length * hop_length
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


if __name__ == "__main__":
    cli_main()
