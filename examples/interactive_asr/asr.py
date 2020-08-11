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

from fairseq import options

from interactive_asr.utils import (
    add_asr_eval_argument,
    setup_asr,
    get_microphone_transcription,
    transcribe_file,
)


def main(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    task, generator, models, sp, tgt_dict = setup_asr(args, logger)

    print("READY!")
    if args.input_file:
        transcription_time, transcription = transcribe_file(
            args, task, generator, models, sp, tgt_dict
        )
        print("transcription:", transcription)
        print("transcription_time:", transcription_time)
    else:
        for transcription in get_microphone_transcription(
            args, task, generator, models, sp, tgt_dict
        ):
            print(
                "{}: {}".format(
                    dt.datetime.now().strftime("%H:%M:%S"), transcription[0][0]
                )
            )


def cli_main():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
