import os

import torch

from cffi import FFI

ffi = FFI()
from ._ext import th_sox


def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError('Expected a tensor, got %s' % type(src))
    if not src.__module__ == 'torch':
        raise TypeError('Expected a CPU based tensor, got %s' % type(src))

def load(filepath, out=None, normalization=None):
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))
    # initialize output tensor
    if out is not None:
        check_input(out)
    else:
        out = torch.FloatTensor()
    # load audio signal
    typename = type(out).__name__.replace('Tensor', '')
    func = getattr(th_sox, 'libthsox_{}_read_audio_file'.format(typename))
    sample_rate_p = ffi.new('int*')
    func(str(filepath).encode("utf-8"), out, sample_rate_p)
    sample_rate = sample_rate_p[0]
    # normalize if needed
    if isinstance(normalization, bool) and normalization:
        out /= 1 << 31 # assuming 16-bit depth
    elif isinstance(normalization, (float, int)):
        out /= normalization # normalize with custom value
    return out, sample_rate

def save(filepath, src, sample_rate):
    # check if save directory exists
    abs_dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(abs_dirpath):
        raise OSError("Directory does not exist: {}".format(abs_dirpath))
    # Check/Fix shape of source data
    if len(src.size()) == 1:
        # 1d tensors as assumed to be mono signals
        src.unsqueeze_(1)
    elif len(src.size()) > 2 or src.size(1) > 2:
        raise ValueError("Expected format (L x N), N = 1 or 2, but found {}".format(src.size()))
    # check if sample_rate is an integer
    if not isinstance(sample_rate, int):
        if int(sample_rate) == sample_rate:
            sample_rate = int(sample_rate)
        else:
            raise TypeError('Sample rate should be a integer')
    # programs such as librosa normalize the signal, unnormalize if detected
    if src.min() >= -1.0 and src.max() <= 1.0:
        src = src * (1 << 31) # assuming 16-bit depth
        src = src.long()
    # save data to file
    filename, extension = os.path.splitext(filepath)
    check_input(src)
    typename = type(src).__name__.replace('Tensor', '')
    func = getattr(th_sox, 'libthsox_{}_write_audio_file'.format(typename))
    func(bytes(filepath, "utf-8"), src, bytes(extension[1:], "utf-8"), sample_rate)
