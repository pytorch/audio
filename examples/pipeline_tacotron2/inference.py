from functools import partial
import argparse
import os
import random
import sys

import torch
import torchaudio
import numpy as np
from torchaudio.prototype.tacotron2 import Tacotron2
from torchaudio.models.wavernn import _MODEL_CONFIG_AND_URLS

from utils import prepare_input_sequence
from datasets import InverseSpectralNormalization
from text.text_preprocessing import (
    available_symbol_set,
    available_phonemizers,
    get_symbol_list,
    text_to_sequence,
)


def parse_args(parser):
    r"""
    Parse commandline arguments.
    """
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        required=True,
        help='[string] Path to the checkpoint file.'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default="./audio.wav",
        help='[string] Path to the output .wav file.'
    )
    parser.add_argument(
        '--input-text',
        '-i',
        type=str,
        default="Hello world",
        help='[string] Type in something here and TTS will generate it!'
    )
    parser.add_argument(
        '--vocoder',
        default='nvidia_waveglow',
        choices=['griffin_lim', 'wavernn', 'nvidia_waveglow'],
        type=str,
        help="Select the vocoder to use.",
    )
    parser.add_argument(
        "--jit",
        default=False,
        action="store_true",
        help="If used, the model and inference function is jitted."
    )

    preprocessor = parser.add_argument_group('text preprocessor setup')
    preprocessor.add_argument(
        '--text-preprocessor',
        default='english_characters',
        type=str,
        choices=available_symbol_set,
        help='select text preprocessor to use.'
    )
    preprocessor.add_argument(
        '--phonemizer', 
        default="DeepPhonemizer",
        type=str,
        choices=available_phonemizers,
        help='select phonemizer to use, only used when text-preprocessor is "english_phonemes"'
    )
    preprocessor.add_argument(
        '--phonemizer-checkpoint',
        default="./en_us_cmudict_forward.pt",
        type=str,
        help='the path or name of the checkpoint for the phonemizer, '
             'only used when text-preprocessor is "english_phonemes"'
    )
    preprocessor.add_argument(
        '--cmudict-root',
        default="./",
        type=str,
        help='the root directory for storing CMU dictionary files'
    )

    audio = parser.add_argument_group('audio parameters')
    audio.add_argument(
        '--sample-rate',
        default=22050,
        type=int,
        help='Sampling rate'
    )
    audio.add_argument(
        '--n-fft',
        default=1024,
        type=int,
        help='Filter length for STFT'
    )
    audio.add_argument(
        '--n-mels',
        default=80,
        type=int,
        help=''
    )
    audio.add_argument(
        '--mel-fmin',
        default=0.0,
        type=float,
        help='Minimum mel frequency'
    )
    audio.add_argument(
        '--mel-fmax',
        default=8000.0,
        type=float,
        help='Maximum mel frequency'
    )

    # parameters for WaveRNN
    wavernn = parser.add_argument_group('WaveRNN parameters')
    wavernn.add_argument(
        '--wavernn-checkpoint-name',
        default="wavernn_10k_epochs_8bits_ljspeech",
        choices=list(_MODEL_CONFIG_AND_URLS.keys()),
        help="Select the WaveRNN checkpoint."
    )
    wavernn.add_argument(
        "--wavernn-loss",
        default="crossentropy",
        choices=["crossentropy"],
        type=str,
        help="The type of loss the WaveRNN pretrained model is trained on.",
    )
    wavernn.add_argument(
        "--wavernn-no-batch-inference",
        default=False,
        action="store_true",
        help="Don't use batch inference for WaveRNN inference."
    )
    wavernn.add_argument(
        "--wavernn-no-mulaw",
        default=False,
        action="store_true",
        help="Don't use mulaw decoder to decode the signal."
    )
    wavernn.add_argument(
        "--wavernn-batch-timesteps",
        default=11000,
        type=int,
        help="The time steps for each batch. Only used when batch inference is used",
    )
    wavernn.add_argument(
        "--wavernn-batch-overlap",
        default=550,
        type=int,
        help="The overlapping time steps between batches. Only used when batch inference is used",
    )

    return parser


def unwrap_distributed(state_dict):
    r"""torch.distributed.DistributedDataParallel wraps the model with an additional "module.".
    This function unwraps this layer so that the weights can be loaded on models with a single GPU.

    Args:
        state_dict: Original state_dict.

    Return:
        unwrapped_state_dict: Unwrapped state_dict.
    """

    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def main(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_symbols = len(get_symbol_list(args.text_preprocessor))
    text_preprocessor = partial(
        text_to_sequence,
        symbol_list=args.text_preprocessor,
        phonemizer=args.phonemizer,
        checkpoint=args.phonemizer_checkpoint,
        cmudict_root=args.cmudict_root,
    )

    tacotron2 = Tacotron2(n_symbol=n_symbols)
    tacotron2.load_state_dict(
        unwrap_distributed(torch.load(args.checkpoint_path, map_location=device)['state_dict']))
    tacotron2 = tacotron2.to(device).eval()

    if args.jit:
        tacotron2 = torch.jit.script(tacotron2)

    sequences, lengths = prepare_input_sequence([args.input_text],
                                                text_processor=text_preprocessor)
    sequences, lengths = sequences.long().to(device), lengths.long().to(device)
    with torch.no_grad():
        mel_specgram, _, _ = tacotron2.infer(sequences, lengths)

    if args.vocoder == "nvidia_waveglow":
        waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(device)
        waveglow.eval()

        if args.jit:
            raise ValueError("Vocoder option `nvidia_waveglow is not jittable.")

        with torch.no_grad():
            waveform = waveglow.infer(mel_specgram).cpu()

    elif args.vocoder == "wavernn":
        from torchaudio.models import wavernn
        sys.path.append(os.path.join(os.path.dirname(__file__), "../pipeline_wavernn"))
        from wavernn_inference_wrapper import WaveRNNInferenceWrapper
        from processing import NormalizeDB

        wavernn_model = wavernn(args.wavernn_checkpoint_name).eval().to(device)
        wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model)

        if args.jit:
            wavernn_inference_model = torch.jit.script(wavernn_inference_model)

        # WaveRNN spectro setting for default checkpoint
        # n_fft = 2048
        # n_mels = 80
        # win_length = 1100
        # hop_length = 275
        # f_min = 40
        # f_max = 11025

        transforms = torch.nn.Sequential(
            InverseSpectralNormalization(),
            NormalizeDB(min_level_db=-100, normalization=True),
        )
        mel_specgram = transforms(mel_specgram.cpu())

        with torch.no_grad():
            waveform = wavernn_inference_model(mel_specgram.to(device),
                                               loss_name=args.wavernn_loss,
                                               mulaw=(not args.wavernn_no_mulaw),
                                               batched=(not args.wavernn_no_batch_inference),
                                               timesteps=args.wavernn_batch_timesteps,
                                               overlap=args.wavernn_batch_overlap,)
        waveform = waveform.unsqueeze(0)

    elif args.vocoder == "griffin_lim":
        from torchaudio.transforms import GriffinLim, InverseMelScale

        inv_norm = InverseSpectralNormalization()
        inv_mel = InverseMelScale(
            n_stft=(args.n_fft // 2 + 1),
            n_mels=args.n_mels,
            sample_rate=args.sample_rate,
            f_min=args.mel_fmin,
            f_max=args.mel_fmax,
            mel_scale="slaney",
            norm='slaney',
        )
        griffin_lim = GriffinLim(
            n_fft=args.n_fft,
            power=1,
            hop_length=256,
            win_length=1024,
        )

        vocoder = torch.nn.Sequential(
            inv_norm,
            inv_mel,
            griffin_lim
        )

        if args.jit:
            vocoder = torch.jit.script(vocoder)

        waveform = vocoder(mel_specgram.cpu())

    torchaudio.save(args.output_path, waveform, args.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    main(args)
