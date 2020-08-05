import os
import tempfile
from typing import Optional
from packaging import version

import torch
import torchaudio

_MIN_VER = version.parse("0.6.0a0")
_MAX_VER = version.parse("0.7.0")
_RUNTIME_VER = version.parse(torchaudio.__version__)


def info(filepath: str) -> torchaudio.backend.sox_io_backend.AudioMetaData:
    return torchaudio.info(filepath)


def load(
    filepath: str,
    frame_offset: int,
    num_frames: int,
    normalize: bool,
    channels_first: bool,
):
    return torchaudio.load(
        filepath, frame_offset, num_frames, normalize, channels_first
    )


def save(
    filepath: str,
    tensor: torch.Tensor,
    sample_rate: int,
    channels_first: bool = True,
    compression: Optional[float] = None,
):
    torchaudio.save(filepath, tensor, sample_rate, channels_first, compression)


def generate(output_dir):
    if not (_MIN_VER <= _RUNTIME_VER < _MAX_VER):
        raise RuntimeError(f"Invalid torchaudio runtime version: {_RUNTIME_VER}")

    torchaudio.set_audio_backend("sox_io")

    funcs = [
        info,
        load,
        save,
    ]

    os.makedirs(output_dir, exist_ok=True)
    for func in funcs:
        torch.jit.script(func).save(os.path.join(output_dir, f"{func.__name__}.zip"))


def validate(input_dir):
    torchaudio.set_audio_backend("sox_io")

    # See https://github.com/pytorch/pytorch/issues/42258
    # info_ = torch.jit.load(os.path.join(input_dir, 'info.zip'))
    load_ = torch.jit.load(os.path.join(input_dir, "load.zip"))
    save_ = torch.jit.load(os.path.join(input_dir, "save.zip"))

    sample_rate = 44100
    normalize = True
    channels_first = True
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "test.wav")
        temp_data = torch.rand(2, sample_rate, dtype=torch.float32)

        save_(temp_file, temp_data, sample_rate, channels_first, 0.0)
        # info_(temp_file)
        load_(temp_file, 0, -1, normalize, channels_first)
