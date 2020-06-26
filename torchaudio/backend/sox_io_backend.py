from typing import Tuple

import torch
from torchaudio._internal import (
    module_utils as _mod_utils,
)


@_mod_utils.requires_module('torchaudio._torchaudio')
def info(filepath: str) -> torch.classes.torchaudio.SignalInfo:
    """Get signal information of an audio file."""
    return torch.ops.torchaudio.sox_io_get_info(filepath)


@_mod_utils.requires_module('torchaudio._torchaudio')
def load(
        filepath: str,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio data from file.

    This function can handle all the codecs that underlying libsox can handle, however note the
    followings.

    Note 1:
        Current torchaudio's binary release only contains codecs for MP3, FLAC and OGG/VORBIS.
        If you need other formats, you need to build torchaudio from source with libsox and
        the corresponding codecs. Refer to README for this.

    Note 2:
        This function is tested on the following formats;
         - WAV
            - 32-bit floating-point
            - 32-bit signed integer
            - 16-bit signed integer
            -  8-bit unsigned integer
         - MP3
         - FLAC
         - OGG/VORBIS

    By default, this function returns Tensor with ``float32`` dtype and the shape of ``[channel, time]``.
    The samples are normalized to fit in the range of ``[-1.0, 1.0]``.

    When the input format is WAV with integer type, such as 32-bit signed integer, 16-bit
    signed integer and 8-bit unsigned integer (24-bit signed integer is not supported),
    by providing ``normalize=False``, this function can return integer Tensor, where the samples
    are expressed within the whole range of the corresponding dtype, that is, ``int32`` tensor
    for 32-bit signed PCM, ``int16`` for 16-bit signed PCM and ``uint8`` for 8-bit unsigned PCM.

    ``normalize`` parameter has no effect on 32-bit floating-point WAV and other formats, such as
    flac and mp3. For these formats, this function always returns ``float32`` Tensor with values
    normalized  to ``[-1.0, 1.0]``.

    Args:
        filepath: Path to audio file
        frame_offset: Number of frames to skip before start reading data.
        num_frames: Maximum number of frames to read. -1 reads all the remaining samples, starting
            from ``frame_offset``. This function may return the less number of frames if there is
            not enough frames in the given file.
        normalize: When ``True``, this function always return ``float32``, and sample values are
            normalized to ``[-1.0, 1.0]``. If input file is integer WAV, giving ``False`` will change
            the resulting Tensor type to integer type. This argument has no effect for formats other
            than integer WAV type.
        channels_first: When True, the returned Tensor has dimension ``[channel, time]``.
            Otherwise, the returned Tensor's dimension is ``[time, channel]``.

    Returns:
        torch.Tensor: If the input file has integer wav format and normalization is off, then it has
            integer type, else ``float32`` type. If ``channels_first=True``, it has
            ``[channel, time]`` else ``[time, channel]``.
    """
    signal = torch.ops.torchaudio.sox_io_load_audio_file(
        filepath, frame_offset, num_frames, normalize, channels_first)
    return signal.get_tensor(), signal.get_sample_rate()


load_wav = load
