from argparse import ArgumentParser
import pathlib
import torch
import torchaudio
from torchaudio.prototype.rnnt_pipeline import EMFORMER_RNNT_BASE_LIBRISPEECH


SAMPLES_PER_CHUNK = 640


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--librispeech_path", type=pathlib.Path, help="Path to LibriSpeech datasets.",
    )
    args = parser.parse_args()

    dataset = torchaudio.datasets.LIBRISPEECH(args.librispeech_path, url="test-clean")
    decoder = EMFORMER_RNNT_BASE_LIBRISPEECH.get_decoder()
    token_processor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_token_processor()
    feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_feature_extractor()
    streaming_feature_extractor = EMFORMER_RNNT_BASE_LIBRISPEECH.get_streaming_feature_extractor()

    for idx in range(10):
        sample = dataset[idx]

        with torch.no_grad():
            waveform = sample[0].squeeze()

            # Streaming decode.
            state, hypothesis = None, None
            for idx in range(0, len(waveform), 4 * SAMPLES_PER_CHUNK):
                segment = waveform[idx: idx + 5 * SAMPLES_PER_CHUNK]
                segment = torch.nn.functional.pad(segment, (0, 5 * SAMPLES_PER_CHUNK - len(segment)))
                features, length = streaming_feature_extractor(segment)
                hypos, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
                hypothesis = hypos[0]
                transcript = token_processor(hypothesis.tokens)
                if transcript:
                    print(transcript, end=" ", flush=True)
            print()

            # Non-streaming decode.
            features, length = feature_extractor(waveform)
            hypos = decoder(features, length, 10)
            print(token_processor(hypos[0].tokens))


if __name__ == "__main__":
    cli_main()
