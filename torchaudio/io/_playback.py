import warnings
from sys import platform
from typing import Optional

import torch
import torchaudio

dict_format = {
    torch.uint8: "u8",
    torch.int16: "s16",
    torch.int32: "s32",
    torch.int64: "s64",
    torch.float32: "flt",
    torch.float64: "dbl",
}


@torchaudio._extension.fail_if_no_ffmpeg
def play_audio(
    waveform: torch.Tensor,
    sample_rate: Optional[float],
    device: Optional[str] = None,
) -> None:
    """Plays audio through specified or available output device.

    .. warning::
       This function is currently only supported on MacOS, and requires
       libavdevice (FFmpeg) with ``audiotoolbox`` output device.

    .. note::
       This function can play up to two audio channels.

    Args:
        waveform: Tensor containing the audio to play.
            Expected shape: `(time, num_channels)`.
        sample_rate: Sample rate of the audio to play.
        device: Output device to use. If None, the default device is used.
    """

    if platform == "darwin":
        device = device or "audiotoolbox"
        path = "-"
    else:
        raise ValueError(f"This function only supports MacOS, but current OS is {platform}")

    available_devices = list(torchaudio.utils.ffmpeg_utils.get_output_devices().keys())
    if device not in available_devices:
        raise ValueError(f"Device {device} is not available. Available devices are: {available_devices}")

    if waveform.dtype not in dict_format:
        raise ValueError(f"Unsupported type {waveform.dtype}. The list of supported types is: {dict_format.keys()}")
    format = dict_format[waveform.dtype]

    if waveform.ndim != 2:
        raise ValueError(f"Expected 2D tensor with shape `(time, num_channels)`, got {waveform.ndim}D tensor instead")

    time, num_channels = waveform.size()
    if num_channels > 2:
        warnings.warn(
            f"Expected up to 2 channels, got {num_channels} channels instead. Only the first 2 channels will be played."
        )

    # Write to speaker device
    s = torchaudio.io.StreamWriter(dst=path, format=device)
    s.add_audio_stream(sample_rate, num_channels, format=format)

    # write audio to the device
    block_size = 256
    with s.open():
        for i in range(0, time, block_size):
            s.write_audio_chunk(0, waveform[i : i + block_size, :])
