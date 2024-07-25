#!/usr/bin/env python3
"""
Create a data preprocess pipeline that can be run with libtorchaudio
"""
import argparse
import os

import torch
import torchaudio


class Pipeline(torch.nn.Module):
    """Example audio process pipeline.

    This example load waveform from a file then apply effects and save it to a file.
    """

    def __init__(self, rir_path: str):
        super().__init__()
        rir, sample_rate = torchaudio.load(rir_path)
        self.register_buffer("rir", rir)
        self.rir_sample_rate: int = sample_rate

    def forward(self, input_path: str, output_path: str):
        torchaudio.sox_effects.init_sox_effects()

        # 1. load audio
        waveform, sample_rate = torchaudio.load(input_path)

        # 2. Add background noise
        alpha = 0.01
        waveform = alpha * torch.randn_like(waveform) + (1 - alpha) * waveform

        # 3. Reample the RIR filter to much the audio sample rate
        rir, _ = torchaudio.sox_effects.apply_effects_tensor(
            self.rir, self.rir_sample_rate, effects=[["rate", str(sample_rate)]]
        )
        rir = rir / torch.linalg.vector_norm(rir, ord=2)
        rir = torch.flip(rir, [1])

        # 4. Apply RIR filter
        waveform = torch.nn.functional.pad(waveform, (rir.shape[1] - 1, 0))
        waveform = torch.nn.functional.conv1d(waveform[None, ...], rir[None, ...])[0]

        # Save
        torchaudio.save(output_path, waveform, sample_rate)


def _create_jit_pipeline(rir_path, output_path):
    module = torch.jit.script(Pipeline(rir_path))
    print("*" * 40)
    print("* Pipeline code")
    print("*" * 40)
    print()
    print(module.code)
    print("*" * 40)
    module.save(output_path)


def _get_path(*paths):
    return os.path.join(os.path.dirname(__file__), *paths)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rir-path", default=_get_path("..", "data", "rir.wav"), help="Audio dara for room impulse response."
    )
    parser.add_argument("--output-path", default=_get_path("pipeline.zip"), help="Output JIT file.")
    return parser.parse_args()


def _main():
    args = _parse_args()
    _create_jit_pipeline(args.rir_path, args.output_path)


if __name__ == "__main__":
    _main()
