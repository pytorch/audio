import argparse

import torch
import torchaudio
from processing import NormalizeDB
from torchaudio.datasets import LJSPEECH
from torchaudio.models import wavernn
from torchaudio.models.wavernn import _MODEL_CONFIG_AND_URLS
from torchaudio.transforms import MelSpectrogram
from wavernn_inference_wrapper import WaveRNNInferenceWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-wav-path",
        default="./output.wav",
        type=str,
        metavar="PATH",
        help="The path to output the reconstructed wav file.",
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="If used, the model and inference function is jitted."
    )
    parser.add_argument("--no-batch-inference", default=False, action="store_true", help="Don't use batch inference.")
    parser.add_argument(
        "--no-mulaw", default=False, action="store_true", help="Don't use mulaw decoder to decoder the signal."
    )
    parser.add_argument(
        "--checkpoint-name",
        default="wavernn_10k_epochs_8bits_ljspeech",
        choices=list(_MODEL_CONFIG_AND_URLS.keys()),
        help="Select the WaveRNN checkpoint.",
    )
    parser.add_argument(
        "--batch-timesteps",
        default=100,
        type=int,
        help="The time steps for each batch. Only used when batch inference is used",
    )
    parser.add_argument(
        "--batch-overlap",
        default=5,
        type=int,
        help="The overlapping time steps between batches. Only used when batch inference is used",
    )
    args = parser.parse_args()
    return args


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    waveform, sample_rate, _, _ = LJSPEECH("./", download=True)[0]

    mel_kwargs = {
        "sample_rate": sample_rate,
        "n_fft": 2048,
        "f_min": 40.0,
        "n_mels": 80,
        "win_length": 1100,
        "hop_length": 275,
        "mel_scale": "slaney",
        "norm": "slaney",
        "power": 1,
    }
    transforms = torch.nn.Sequential(
        MelSpectrogram(**mel_kwargs),
        NormalizeDB(min_level_db=-100, normalization=True),
    )
    mel_specgram = transforms(waveform)

    wavernn_model = wavernn(args.checkpoint_name).eval().to(device)
    wavernn_inference_model = WaveRNNInferenceWrapper(wavernn_model)

    if args.jit:
        wavernn_inference_model = torch.jit.script(wavernn_inference_model)

    with torch.no_grad():
        output = wavernn_inference_model(
            mel_specgram.to(device),
            mulaw=(not args.no_mulaw),
            batched=(not args.no_batch_inference),
            timesteps=args.batch_timesteps,
            overlap=args.batch_overlap,
        )

    torchaudio.save(args.output_wav_path, output, sample_rate=sample_rate)


if __name__ == "__main__":
    args = parse_args()
    main(args)
