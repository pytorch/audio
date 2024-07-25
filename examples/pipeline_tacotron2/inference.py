"""
Text-to-speech pipeline using Tacotron2.
"""

import argparse
import os
import random
import sys
from functools import partial

import numpy as np
import torch
import torchaudio
from datasets import InverseSpectralNormalization
from text.text_preprocessing import available_phonemizers, available_symbol_set, get_symbol_list, text_to_sequence
from torchaudio.models import Tacotron2, tacotron2 as pretrained_tacotron2
from utils import prepare_input_sequence


def parse_args():
    r"""
    Parse commandline arguments.
    """
    from torchaudio.models.tacotron2 import _MODEL_CONFIG_AND_URLS as tacotron2_config_and_urls
    from torchaudio.models.wavernn import _MODEL_CONFIG_AND_URLS as wavernn_config_and_urls

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        choices=list(tacotron2_config_and_urls.keys()),
        help="[string] The name of the checkpoint to load.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="[string] Path to the checkpoint file.")
    parser.add_argument("--output-path", type=str, default="./audio.wav", help="[string] Path to the output .wav file.")
    parser.add_argument(
        "--input-text",
        "-i",
        type=str,
        default="Hello world",
        help="[string] Type in something here and TTS will generate it!",
    )
    parser.add_argument(
        "--vocoder",
        default="nvidia_waveglow",
        choices=["griffin_lim", "wavernn", "nvidia_waveglow"],
        type=str,
        help="Select the vocoder to use.",
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="If used, the model and inference function is jitted."
    )

    preprocessor = parser.add_argument_group("text preprocessor setup")
    preprocessor.add_argument(
        "--text-preprocessor",
        default="english_characters",
        type=str,
        choices=available_symbol_set,
        help="select text preprocessor to use.",
    )
    preprocessor.add_argument(
        "--phonemizer",
        default="DeepPhonemizer",
        type=str,
        choices=available_phonemizers,
        help='select phonemizer to use, only used when text-preprocessor is "english_phonemes"',
    )
    preprocessor.add_argument(
        "--phonemizer-checkpoint",
        default="./en_us_cmudict_forward.pt",
        type=str,
        help="the path or name of the checkpoint for the phonemizer, "
        'only used when text-preprocessor is "english_phonemes"',
    )
    preprocessor.add_argument(
        "--cmudict-root", default="./", type=str, help="the root directory for storing CMU dictionary files"
    )

    audio = parser.add_argument_group("audio parameters")
    audio.add_argument("--sample-rate", default=22050, type=int, help="Sampling rate")
    audio.add_argument("--n-fft", default=1024, type=int, help="Filter length for STFT")
    audio.add_argument("--n-mels", default=80, type=int, help="")
    audio.add_argument("--mel-fmin", default=0.0, type=float, help="Minimum mel frequency")
    audio.add_argument("--mel-fmax", default=8000.0, type=float, help="Maximum mel frequency")

    # parameters for WaveRNN
    wavernn = parser.add_argument_group("WaveRNN parameters")
    wavernn.add_argument(
        "--wavernn-checkpoint-name",
        default="wavernn_10k_epochs_8bits_ljspeech",
        choices=list(wavernn_config_and_urls.keys()),
        help="Select the WaveRNN checkpoint.",
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
        help="Don't use batch inference for WaveRNN inference.",
    )
    wavernn.add_argument(
        "--wavernn-no-mulaw", default=False, action="store_true", help="Don't use mulaw decoder to decode the signal."
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

    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def nvidia_waveglow_vocode(mel_specgram, device, jit=False):
    waveglow = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_waveglow", model_math="fp16")
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to(device)
    waveglow.eval()

    if args.jit:
        raise ValueError("Vocoder option `nvidia_waveglow is not jittable.")

    with torch.no_grad():
        waveform = waveglow.infer(mel_specgram).cpu()

    return waveform


def wavernn_vocode(
    mel_specgram,
    wavernn_checkpoint_name,
    wavernn_loss,
    wavernn_no_mulaw,
    wavernn_no_batch_inference,
    wavernn_batch_timesteps,
    wavernn_batch_overlap,
    device,
    jit,
):
    from torchaudio.models import wavernn

    sys.path.append(os.path.join(os.path.dirname(__file__), "../pipeline_wavernn"))
    from processing import NormalizeDB
    from wavernn_inference_wrapper import WaveRNNInferenceWrapper

    wavernn_model = wavernn(wavernn_checkpoint_name).eval().to(device)
    wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model)

    if jit:
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
        waveform = wavernn_inference_model(
            mel_specgram.to(device),
            loss_name=wavernn_loss,
            mulaw=(not wavernn_no_mulaw),
            batched=(not wavernn_no_batch_inference),
            timesteps=wavernn_batch_timesteps,
            overlap=wavernn_batch_overlap,
        )
    return waveform.unsqueeze(0)


def griffin_lim_vocode(
    mel_specgram,
    n_fft,
    n_mels,
    sample_rate,
    mel_fmin,
    mel_fmax,
    jit,
):
    from torchaudio.transforms import GriffinLim, InverseMelScale

    inv_norm = InverseSpectralNormalization()
    inv_mel = InverseMelScale(
        n_stft=(n_fft // 2 + 1),
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=mel_fmin,
        f_max=mel_fmax,
        mel_scale="slaney",
        norm="slaney",
    )
    griffin_lim = GriffinLim(
        n_fft=n_fft,
        power=1,
        hop_length=256,
        win_length=1024,
    )

    vocoder = torch.nn.Sequential(inv_norm, inv_mel, griffin_lim)

    if jit:
        vocoder = torch.jit.script(vocoder)

    waveform = vocoder(mel_specgram.cpu())
    return waveform


def main(args):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.checkpoint_path is None and args.checkpoint_name is None:
        raise ValueError("Either --checkpoint-path or --checkpoint-name must be specified.")
    elif args.checkpoint_path is not None and args.checkpoint_name is not None:
        raise ValueError("Both --checkpoint-path and --checkpoint-name are specified, " "can only specify one.")

    n_symbols = len(get_symbol_list(args.text_preprocessor))
    text_preprocessor = partial(
        text_to_sequence,
        symbol_list=args.text_preprocessor,
        phonemizer=args.phonemizer,
        checkpoint=args.phonemizer_checkpoint,
        cmudict_root=args.cmudict_root,
    )

    if args.checkpoint_path is not None:
        tacotron2 = Tacotron2(n_symbol=n_symbols)
        tacotron2.load_state_dict(
            unwrap_distributed(torch.load(args.checkpoint_path, map_location=device)["state_dict"])
        )
        tacotron2 = tacotron2.to(device).eval()
    elif args.checkpoint_name is not None:
        tacotron2 = pretrained_tacotron2(args.checkpoint_name).to(device).eval()

        if n_symbols != tacotron2.n_symbols:
            raise ValueError(
                "the number of symbols for text_preprocessor ({n_symbols}) "
                "should match the number of symbols for the"
                "pretrained tacotron2 ({tacotron2.n_symbols})."
            )

    if args.jit:
        tacotron2 = torch.jit.script(tacotron2)

    sequences, lengths = prepare_input_sequence([args.input_text], text_processor=text_preprocessor)
    sequences, lengths = sequences.long().to(device), lengths.long().to(device)
    with torch.no_grad():
        mel_specgram, _, _ = tacotron2.infer(sequences, lengths)

    if args.vocoder == "nvidia_waveglow":
        waveform = nvidia_waveglow_vocode(mel_specgram=mel_specgram, device=device, jit=args.jit)

    elif args.vocoder == "wavernn":
        waveform = wavernn_vocode(
            mel_specgram=mel_specgram,
            wavernn_checkpoint_name=args.wavernn_checkpoint_name,
            wavernn_loss=args.wavernn_loss,
            wavernn_no_mulaw=args.wavernn_no_mulaw,
            wavernn_no_batch_inference=args.wavernn_no_batch_inference,
            wavernn_batch_timesteps=args.wavernn_batch_timesteps,
            wavernn_batch_overlap=args.wavernn_batch_overlap,
            device=device,
            jit=args.jit,
        )

    elif args.vocoder == "griffin_lim":
        waveform = griffin_lim_vocode(
            mel_specgram=mel_specgram,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            sample_rate=args.sample_rate,
            mel_fmin=args.mel_fmin,
            mel_fmax=args.mel_fmax,
            jit=args.jit,
        )

    torchaudio.save(args.output_path, waveform, args.sample_rate)


if __name__ == "__main__":
    parser = parse_args()
    args, _ = parser.parse_known_args()

    main(args)
