#!/usr/bin/env python3
"""Evaluate ASR model with LibriSpeech dataset"""

import logging

logger = logging.getLogger(__name__)


def _get_model(model):
    import torchaudio

    bundle = getattr(torchaudio.pipelines, model)
    return bundle.get_model()


def _get_model_espnet(model):
    import torch
    from espnet2.bin.asr_inference import Speech2Text

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, speech):
            length = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
            return self.model.asr_model.encode(speech=speech, speech_lengths=length)

    model = Speech2Text.from_pretrained(model)
    return Wrapper(model)


def run_inference(args):
    import torch
    import torchaudio
    from torchaudio.models.decoder import download_pretrained_files, ctc_decoder

    if args.model_source == "torchaudio":
        model = _get_model(args.model)
    else:
        model = _get_model_espnet(args.model)

    # get decoder files
    files = download_pretrained_files("librispeech-4-gram")

    decoder = ctc_decoder(
        lexicon=files.lexicon,
        tokens=files.tokens,
        lm=files.lm,
        nbest=args.nbest,
        beam_size=args.beam_size,
        beam_size_token=args.beam_size_token,
        beam_threshold=args.beam_threshold,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        unk_score=args.unk_score,
        sil_score=args.sil_score,
        log_add=False,
    )

    dataset = torchaudio.datasets.LIBRISPEECH(args.librispeech_path, url=args.split, download=False)

    total_edit_distance = 0
    total_length = 0
    for idx, sample in enumerate(dataset):
        waveform, _, transcript, _, _, _ = sample
        transcript = transcript.strip().lower().strip()

        with torch.inference_mode():
            emission, _ = model(waveform)
        results = decoder(emission)

        total_edit_distance += torchaudio.functional.edit_distance(transcript.split(), results[0][0].words)
        total_length += len(transcript.split())

        if idx % 100 == 0:
            logger.info(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.info(f"Final WER: {total_edit_distance / total_length}")


def _parse_args():
    import argparse
    from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter

    class Formatter(RawTextHelpFormatter, ArgumentDefaultsHelpFormatter):
        pass

    # fmt: off
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=Formatter,
    )
    parser.add_argument(
        "--librispeech-path",
        required=True,
        help="folder where LibriSpeech is stored",
    )
    parser.add_argument(
        "--split",
        help="LibriSpeech dataset split",
        choices=["dev-clean", "dev-other", "test-clean", "test-other"],
        default="test-other",
    )
    parser.add_argument(
        "--model",
        default="WAV2VEC2_ASR_BASE_960H",
        help="pretrained Wav2Vec2 model from torchaudio.pipelines",
    )
    parser.add_argument(
        "--model-source",
        default="torchaudio",
        choices=["torchaudio", "espnet2"],
        help="project in which the model is defined",
    )
    parser.add_argument(
        "--nbest",
        type=int,
        default=1,
        help="number of best hypotheses to return",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=500,
        help="beam size for determining number of hypotheses to store",
    )
    parser.add_argument(
        "--beam-size-token",
        type=int,
        default=None,
        help="number of tokens to consider at each beam search step",
    )
    parser.add_argument(
        "--beam-threshold",
        type=int,
        default=50,
        help="beam threshold for pruning hypotheses",
    )
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
    parser.add_argument(
        "--unk-score",
        type=float,
        default=float("-inf"),
        help="unknown word insertion score",
    )
    parser.add_argument(
        "--sil-score",
        type=float,
        default=0,
        help="silence insertion score",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to use debug level for logging",
    )
    # fmt: on
    return parser.parse_args()


def _init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def _main():
    args = _parse_args()
    _init_logger(args.debug)
    run_inference(args)


if __name__ == "__main__":
    _main()
