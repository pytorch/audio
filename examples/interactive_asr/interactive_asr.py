#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Run inference for pre-processed data with a trained model.
"""

import datetime as dt
import logging
import os
import random
import string
import sys
from time import time

import numpy as np
import torch

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import sentencepiece as spm
import torchaudio
from fairseq import options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module
from vad import get_microphone_chunks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_asr_eval_argument(parser):
    parser.add_argument("--ctc", action="store_true", help="decode a ctc model")
    parser.add_argument("--rnnt", default=False, help="decode a rnnt model")
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )
    parser.add_argument(
        "--lm_weight",
        default=0.2,
        help="weight for wfstlm while interpolating\
with neural score",
    )
    parser.add_argument(
        "--rnnt_len_penalty", default=-0.5, help="rnnt length penalty on word level"
    )
    return parser


def check_args(args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def process_predictions(args, hypos, sp, tgt_dict):
    res = []
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())
        hyp_words = sp.DecodePieces(hyp_pieces.split())
        res.append(hyp_words)
    return res


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation
    """
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


def calc_mean_invstddev(feature):
    if len(feature.shape) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = np.mean(feature, axis=0)
    var = np.var(feature, axis=0)
    # avoid division by ~zero
    if var.any() < sys.float_info.epsilon:
        return mean, 1.0 / (np.sqrt(var) + sys.float_info.epsilon)
    return mean, 1.0 / np.sqrt(var)


def calcMN(features):
    mean, invstddev = calc_mean_invstddev(features)
    res = (features - mean) * invstddev
    return res


def transcribe(waveform, args, task, generator, models, sp, tgt_dict):
    r"""
    CUDA_VISIBLE_DEVICES=0 python infer_asr.py /Users/jamarshon/Documents/downloads/ \
        --task speech_recognition --max-tokens 250000 --nbest 1 --path \
        /Users/jamarshon/Downloads/checkpoint_avg_60_80.pt --beam 20 
    """
    num_features = 80
    output = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=num_features)
    output_cmvn = calcMN(output.cpu().detach().numpy())

    # size (m, n)
    source = torch.tensor(output_cmvn)
    frames_lengths = torch.LongTensor([source.size(0)])

    # size (1, m, n). In general, if source is (x, m, n), then hypos is (x, ...)
    source.unsqueeze_(0)
    sample = {"net_input": {"src_tokens": source, "src_lengths": frames_lengths}}

    hypos = task.inference_step(generator, models, sample)

    assert len(hypos) == 1
    transcription = []
    for i in range(len(hypos)):
        # Process top predictions
        hyp_words = process_predictions(args, hypos[i], sp, tgt_dict)
        transcription.append(hyp_words)
    return transcription


def main(args):
    check_args(args)
    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 30000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)

    # Set dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info("| loading model(s) from {}".format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(":"),
        task,
        model_arg_overrides=eval(args.model_overrides),  # noqa
    )
    optimize_models(args, use_cuda, models)

    # Initialize generator
    generator = task.build_generator(args)

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(args.data, "spm.model"))

    print("READY!")
    for (waveform, sample_rate) in get_microphone_chunks():
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(waveform.reshape(1, -1))
        transcription = transcribe(
            waveform, args, task, generator, models, sp, tgt_dict
        )
        print(
            "{}: {}".format(dt.datetime.now().strftime("%H:%M:%S"), transcription[0][0])
        )


def cli_main():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
