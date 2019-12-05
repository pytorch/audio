import os

import torch


def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError("Expected a tensor, got %s" % type(src))
    if src.is_cuda:
        raise TypeError("Expected a CPU based tensor, got %s" % type(src))


def load(
    filepath,
    out=None,
    normalization=True,
    channels_first=True,
    num_frames=0,
    offset=0,
    filetype=None,
    **_,
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
    # TODO remove pysoundfile and call directly soundfile to avoid going through numpy
    if out is not None:
        check_input(out)
        _, sample_rate = soundfile.read(
            filepath, frames=num_frames, start=offset, always_2d=True, out=out
        )
    else:
        out, sample_rate = soundfile.read(
            filepath, frames=num_frames, start=offset, always_2d=True
        )
        out = torch.tensor(out).t()

    # normalize if needed
    # _audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath, src, sample_rate, channels_first=True, **_):
    r"""See torchaudio.save"""

    if channels_first:
        src = src.t()

    import soundfile
    return soundfile.write(filepath, src, sample_rate)


def info(filepath, **_):
    r"""See torchaudio.info"""

    import soundfile
    return soundfile.info(filepath)
