import os.path

import torch

import torchaudio


def load(
    filepath,
    out=None,
    normalization=True,
    channels_first=True,
    num_frames=0,
    offset=0,
    signalinfo=None,
    encodinginfo=None,
    filetype=None,
):
    r"""See torchaudio.load"""

    # stringify if `pathlib.Path` (noop if already `str`)
    filepath = str(filepath)
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    # initialize output tensor
    if out is not None:
        torchaudio.check_input(out)
    else:
        out = torch.FloatTensor()

    if num_frames < -1:
        raise ValueError("Expected value for num_samples -1 (entire file) or >=0")
    if offset < 0:
        raise ValueError("Expected positive offset value")

    import _torch_sox
    sample_rate = _torch_sox.read_audio_file(
        filepath,
        out,
        channels_first,
        num_frames,
        offset,
        signalinfo,
        encodinginfo,
        filetype
    )

    # normalize if needed
    torchaudio._audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath, src, sample_rate, precision=16, channels_first=True):
    r"""See torchaudio.save"""

    si = torchaudio.sox_signalinfo_t()
    ch_idx = 0 if channels_first else 1
    si.rate = sample_rate
    si.channels = 1 if src.dim() == 1 else src.size(ch_idx)
    si.length = src.numel()
    si.precision = precision
    return torchaudio.save_encinfo(filepath, src, channels_first, si)


def info(filepath):
    r"""See torchaudio.info"""

    import _torch_sox
    return _torch_sox.get_info(filepath)
