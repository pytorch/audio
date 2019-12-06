import os

import torch


def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError("Expected a tensor, got %s" % type(src))
    if src.is_cuda:
        raise TypeError("Expected a CPU based tensor, got %s" % type(src))


def load(
    filepath,
    normalization=True,
    channels_first=True,
    num_frames=0,
    offset=0,
    filetype=None,
    **_
):
    r"""See torchaudio.load"""

    # stringify if `pathlib.Path` (noop if already `str`)
    filepath = str(filepath)

    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if num_frames == 0:
        num_frames = -1
    if offset < 0:
        raise ValueError("Expected positive offset value")

    import soundfile

    # initialize output tensor
    # TODO call libsoundfile directly to avoid numpy
    out, sample_rate = soundfile.read(
        filepath, frames=num_frames, start=offset, always_2d=True
    )
    out = torch.from_numpy(out).t()

    if not channels_first:
        out = out.t()

    # normalize if needed
    # _audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath, src, sample_rate, precision=16, channels_first=True, **_):
    r"""See torchaudio.save"""

    ch_idx, len_idx = (0, 1) if channels_first else (1, 0)

    # check if save directory exists
    abs_dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(abs_dirpath):
        raise OSError("Directory does not exist: {}".format(abs_dirpath))
    # check that src is a CPU tensor
    check_input(src)
    # Check/Fix shape of source data
    if src.dim() == 1:
        # 1d tensors as assumed to be mono signals
        src.unsqueeze_(ch_idx)
    elif src.dim() > 2 or src.size(ch_idx) > 16:
        # assumes num_channels < 16
        raise ValueError(
            "Expected format where C < 16, but found {}".format(src.size()))

    if channels_first:
        src = src.t()

    if src.dtype == torch.int64:
        # Soundfile doesn't support int64
        src = src.type(torch.int32)

    precision = "PCM_S8" if precision == 8 else "PCM_" + str(precision)

    import soundfile
    return soundfile.write(filepath, src, sample_rate, precision)


def info(filepath, **_):
    r"""See torchaudio.info"""

    import soundfile
    return soundfile.info(filepath)
