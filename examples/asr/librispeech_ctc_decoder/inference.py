import argparse
from typing import Optional

import torch
import torchaudio
from torchaudio.prototype.ctc_decoder import kenlm_lexicon_decoder


bundle_map = {
    "FT_10M": torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M,
    "FT_100H": torchaudio.pipelines.WAV2VEC2_ASR_BASE_100H,
    "FT_960H": torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H,
}


def _download_files(lexicon_file, kenlm_file):
    torch.hub.download_url_to_file(
        "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/lexicon-librispeech.txt", lexicon_file
    )
    torch.hub.download_url_to_file(
        "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/4-gram-librispeech.bin", kenlm_file
    )


def run_inference(args):
    # get pretrained wav2vec2.0 model
    bundle = bundle_map[args.model]
    model = bundle.get_model()
    tokens = [label.lower() for label in bundle.get_labels()]

    # get decoder files
    hub_dir = torch.hub.get_dir()
    lexicon_file = f"{hub_dir}/lexicon.txt"
    kenlm_file = f"{hub_dir}/kenlm.bin"
    _download_files(lexicon_file, kenlm_file)

    decoder = kenlm_lexicon_decoder(
        lexicon=lexicon_file,
        tokens=tokens,
        kenlm=kenlm_file,
        nbest=1,
        beam_size=1500,
        beam_size_token=None,
        beam_threshold=50,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        unk_score=float("-inf"),
        sil_score=0,
        log_add=False,
    )

    dataset = torchaudio.datasets.LIBRISPEECH(args.librispeech_path, url=args.split, download=False)

    total_edit_distance = 0
    total_length = 0
    for idx, sample in enumerate(dataset):
        waveform, _, transcript, _, _, _ = sample
        transcript = transcript.strip().lower().strip()

        emission, _ = model(waveform)
        results = decoder(emission)
        hypothesis = " ".join(results[0][0].words).lower().strip()

        total_edit_distance += torchaudio.functional.edit_distance(transcript.split(), hypothesis.split())
        total_length += len(transcript.split())

        if idx % 100 == 0:
            print(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    print(f"Final WER: {total_edit_distance / total_length}")


def cli_main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--librispeech_path",
        type=str,
        help="folder where LibriSpeech is stored",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="LibriSpeech dataset split",
        choices=["dev-clean", "dev-other", "test-clean", "test-other"],
        default="test-other",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FT_960H",
        choices=["FT_10M", "FT_100H", "FT_960H"],
        help="pretrained Wav2Vec2 model",
    )
    parser.add_argument("--nbest", type=int, default=1, help="number of best hypotheses to return")
    parser.add_argument(
        "--beam-size", type=int, default=500, help="beam size for determining number of hypotheses to store"
    )
    parser.add_argument(
        "--beam-size-token",
        type=Optional[int],
        default=None,
        help="number of tokens to consider at each beam search step",
    )
    parser.add_argument("--beam-threshold", type=int, default=50, help="beam threshold for pruning hypotheses")
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=1.74,
        help="languge model weight",
    )
    parser.add_argument(
        "--word-score",
        type=float,
        default=0.52,
        help="word insertion score",
    )
    parser.add_argument("--unk_score", type=float, default=float("-inf"), help="unkown word insertion score")
    parser.add_argument("--sil_score", type=float, default=0, help="silence insertion score")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    cli_main()
