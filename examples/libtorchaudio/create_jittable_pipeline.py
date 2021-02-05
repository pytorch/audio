#!/usr/bin/env python3
"""
Create a data preprocess pipeline that can be run with libtorchaudio
"""
import os
import argparse

import torch
import torchaudio


def _get_path(*paths):
    return os.path.join(os.path.dirname(__file__), *paths)


def _load_rir():
    path = _get_path("data", "rir.wav")
    waveform, sample_rate = torchaudio.load(path)
    rir = torch.flip(waveform, [1])
    return rir, sample_rate


class Pipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rir, sample_rate = _load_rir()
        self.register_buffer('rir', rir)
        self.sample_rate: int = sample_rate

    def forward(self, input_path: str, output_path: str):
        torchaudio.sox_effects.init_sox_effects()
        waveform, _ = torchaudio.sox_effects.apply_effects_file(
            input_path, effects=[["rate", str(self.sample_rate)]])

        waveform = torch.nn.functional.pad(waveform, (self.rir.shape[1]-1, 0))
        augmented = torch.nn.functional.conv1d(waveform[None, ...], self.rir[None, ...])[0]

        torchaudio.save(output_path, augmented, self.sample_rate)


def _create_jit_pipeline(output_path):
    module = torch.jit.script(Pipeline())
    print(module.code)
    module.save(output_path)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        default=_get_path("data", "pipeline.zip"),
        help="Output JIT file."
    )
    return parser.parse_args()


def _main():
    args = _parse_args()
    _create_jit_pipeline(args.output_path)


if __name__ == '__main__':
    _main()
