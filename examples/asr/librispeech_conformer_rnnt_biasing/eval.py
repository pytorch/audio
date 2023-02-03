import os
import logging
import pathlib
from argparse import ArgumentParser

import torch
import torchaudio
from lightning import ConformerRNNTModule
from transforms import get_data_module


logger = logging.getLogger()


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def run_eval(args):
    model = ConformerRNNTModule.load_from_checkpoint(
        args.checkpoint_path, sp_model=str(args.sp_model_path), biasing=args.biasing).eval()
    data_module = get_data_module(str(args.librispeech_path), str(args.global_stats_path), str(args.sp_model_path),
                                  biasinglist=args.biasing_list, droprate=args.droprate, maxsize=args.maxsize)

    if args.use_cuda:
        model = model.to(device="cuda")

    total_edit_distance = 0
    total_length = 0
    dataloader = data_module.test_dataloader()
    hypout = []
    refout = []
    with torch.no_grad():
        for idx, (batch, sample) in enumerate(dataloader):
            filename = 'librispeech_clean_100_{}'.format(idx)
            actual = sample[0][2]
            predicted = model(batch)
            hypout.append('{} ({})\n'.format(predicted.upper().strip(), filename))
            refout.append('{} ({})\n'.format(actual.upper().strip(), filename))
            total_edit_distance += compute_word_level_distance(actual, predicted)
            total_length += len(actual.split())
            if idx % 100 == 0:
                logger.warning(f"Processed elem {idx}; WER: {total_edit_distance / total_length}")
    logger.warning(f"Final WER: {total_edit_distance / total_length}")
    with open(os.path.join(args.expdir, 'hyp.trn.txt'), 'w') as fout:
        fout.writelines(hypout)
    with open(os.path.join(args.expdir, 'ref.trn.txt'), 'w') as fout:
        fout.writelines(refout)


def cli_main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
        required=True,
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("global_stats_100.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--librispeech-path",
        type=pathlib.Path,
        help="Path to LibriSpeech datasets.",
        required=True,
    )
    parser.add_argument(
        "--sp-model-path",
        type=pathlib.Path,
        help="Path to SentencePiece model.",
        required=True,
    )
    parser.add_argument(
        "--expdir",
        type=pathlib.Path,
        help="Output path.",
        required=True,
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=False,
        help="Run using CUDA.",
    )
    parser.add_argument(
        "--biasing-list",
        type=str,
        default="",
        help="Path to the biasing list used for inference.",
    )
    parser.add_argument(
        "--droprate",
        type=float,
        default=0.0,
        help="biasing list true entry drop rate",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=0,
        help="biasing list size",
    )
    parser.add_argument(
        "--biasing",
        type=str,
        help="Use biasing",
    )
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    cli_main()
