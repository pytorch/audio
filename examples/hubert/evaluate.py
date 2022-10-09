import argparse
import logging
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.models.decoder import ctc_decoder, CTCDecoder, download_pretrained_files
from utils import _get_id2label

logger = logging.getLogger(__name__)


def _load_checkpoint(checkpoint: str) -> torch.nn.Module:
    model = torchaudio.models.hubert_base(aux_num_out=29)
    checkpoint = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k in state_dict:
        if "model.wav2vec2" in k:
            new_state_dict[k.replace("model.wav2vec2.", "")] = state_dict[k]
        elif "aux" in k:
            new_state_dict[k] = state_dict[k]
    model.load_state_dict(new_state_dict)
    return model


def _viterbi_decode(emission: torch.Tensor, id2token: Dict, blank_idx: int = 0) -> List[str]:
    """Run greedy decoding for ctc outputs.

    Args:
        emission (torch.Tensor): Output of CTC layer. Tensor with dimensions (..., time, num_tokens).
        id2token (Dictionary): The dictionary that maps indices of emission's last dimension
            to the corresponding tokens.

    Returns:
        (List of str): The decoding result. List of string in lower case.
    """
    hypothesis = emission.argmax(-1).unique_consecutive()
    hypothesis = hypothesis[hypothesis != blank_idx]
    hypothesis = "".join(id2token[int(i)] for i in hypothesis).replace("|", " ").strip()
    return hypothesis.split()


def _ctc_decode(emission, decoder: CTCDecoder) -> List[str]:
    """Run CTC decoding with a KenLM language model.

    Args:
        emission (torch.Tensor): Output of CTC layer. Tensor with dimensions `(..., time, num_tokens)`.
        decoder (CTCDecoder): The initialized CTCDecoder.

    Returns:
        (List of str): The decoding result. List of string in lower case.
    """
    hypothesis = decoder(emission)
    hypothesis = hypothesis[0][0].words
    hypothesis = [word for word in hypothesis if word != " "]
    return hypothesis


def run_inference(args):
    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the fine-tuned HuBERTPretrainModel from checkpoint.
    model = _load_checkpoint(args.checkpoint)
    model.eval().to(device)

    if args.use_lm:
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
    else:
        id2token = _get_id2label()

    dataset = torchaudio.datasets.LIBRISPEECH(args.librispeech_path, url=args.split)

    total_edit_distance = 0
    total_length = 0
    for idx, sample in enumerate(dataset):
        waveform, _, transcript, _, _, _ = sample
        transcript = transcript.strip().lower().strip().replace("\n", "")

        with torch.inference_mode():
            emission, _ = model(waveform.to(device))
            emission = F.log_softmax(emission, dim=-1)
        if args.use_lm:
            hypothesis = _ctc_decode(emission.cpu(), decoder)
        else:
            hypothesis = _viterbi_decode(emission, id2token)

        total_edit_distance += torchaudio.functional.edit_distance(hypothesis, transcript.split())
        total_length += len(transcript.split())

        if idx % 100 == 0:
            logger.info(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.info(f"Final WER: {total_edit_distance / total_length}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--librispeech-path",
        type=str,
        help="Folder where LibriSpeech dataset is stored.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev-clean", "dev-other", "test-clean", "test-other"],
        help="LibriSpeech dataset split. (Default: 'test-clean')",
        default="test-clean",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="The checkpoint path of fine-tuned HuBERTPretrainModel.",
    )
    parser.add_argument("--use-lm", action="store_true", help="Whether to use language model for decoding.")
    parser.add_argument("--nbest", type=int, default=1, help="Number of best hypotheses to return.")
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1500,
        help="Beam size for determining number of hypotheses to store. (Default: 1500)",
    )
    parser.add_argument(
        "--beam-size-token",
        type=int,
        default=29,
        help="Number of tokens to consider at each beam search step. (Default: None)",
    )
    parser.add_argument(
        "--beam-threshold", type=int, default=100, help="Beam threshold for pruning hypotheses. (Default: 100)"
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=2.46,
        help="Languge model weight in decoding. (Default: 2.46)",
    )
    parser.add_argument(
        "--word-score",
        type=float,
        default=-0.59,
        help="Word insertion score in decoding. (Default: -0.59)",
    )
    parser.add_argument(
        "--unk-score", type=float, default=float("-inf"), help="Unknown word insertion score. (Default: -inf)"
    )
    parser.add_argument("--sil-score", type=float, default=0, help="Silence insertion score. (Default: 0)")
    parser.add_argument("--use-gpu", action="store_true", help="Whether to use GPU for decoding.")
    parser.add_argument("--debug", action="store_true", help="Whether to use debug level for logging.")
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
