import os

import torch

_subtype_to_precision = {
    'PCM_S8': 8,
    'PCM_16': 16,
    'PCM_24': 24,
    'PCM_32': 32,
    'PCM_U8': 8
}


class SignalInfo:
    def __init__(self, channels=None, rate=None, precision=None, length=None):
        self.channels = channels
        self.rate = rate
        self.precision = precision
        self.length = length


class EncodingInfo:
    def __init__(
            self,
            encoding=None,
            bits_per_sample=None,
            compression=None,
            reverse_bytes=None,
            reverse_nibbles=None,
            reverse_bits=None,
            opposite_endian=None
    ):
        self.encoding = encoding
        self.bits_per_sample = bits_per_sample
        self.compression = compression
        self.reverse_bytes = reverse_bytes
        self.reverse_nibbles = reverse_nibbles
        self.reverse_bits = reverse_bits
        self.opposite_endian = opposite_endian


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
    signalinfo=None,
    encodinginfo=None,
    filetype=None,
):
    r"""See torchaudio.load"""

    assert out is None
    assert normalization
    assert signalinfo is None
    assert encodinginfo is None

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
        filepath, frames=num_frames, start=offset, dtype="float32", always_2d=True
    )
    out = torch.from_numpy(out).t()

    if not channels_first:
        out = out.t()

    # normalize if needed
    # _audio_normalization(out, normalization)

    return out, sample_rate


def save(filepath, src, sample_rate, precision=16, channels_first=True):
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


def info(filepath):
    r"""See torchaudio.info"""

    import soundfile
    sfi = soundfile.info(filepath)

    precision = _subtype_to_precision[sfi.subtype]
    si = SignalInfo(sfi.channels, sfi.samplerate, precision, sfi.frames)
    ei = EncodingInfo(bits_per_sample=precision)
    return si, ei
