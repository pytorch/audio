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


def load(filename, out=None):
    if out is not None:
        check_input(out)
    else:
        out = torch.FloatTensor()
    typename = type(out).__name__.replace('Tensor', '')
    func = getattr(th_sox, 'libthsox_{}_read_audio_file'.format(typename))
    sample_rate_p = ffi.new('int*')
    func(str(filename).encode("ascii"), out, sample_rate_p)
    sample_rate = sample_rate_p[0]
    return out, sample_rate


def save(filepath, src, sample_rate):
    filename, extension = os.path.splitext(filepath)
    if type(sample_rate) != int:
        raise TypeError('Sample rate should be a integer')

    check_input(src)
    typename = type(src).__name__.replace('Tensor', '')
    func = getattr(th_sox, 'libthsox_{}_write_audio_file'.format(typename))

    func(bytes(filepath, "ascii"), src, extension[1:], sample_rate)
