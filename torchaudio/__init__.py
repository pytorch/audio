from __future__ import division

import os

import torch

from cffi import FFI

ffi = FFI()
from ._ext import th_sox

from torchaudio import transforms
from torchaudio import datasets

def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError('Expected a tensor, got %s' % type(src))
    if not src.__module__ == 'torch':
        raise TypeError('Expected a CPU based tensor, got %s' % type(src))

def get_info(filepath):
    """Returns a dict of information about the audio file at filepath.
    """
    bits_per_sample_p = ffi.new('int *')
    length_p = ffi.new('unsigned long *')
    sample_rate_p = ffi.new('unsigned int *')
    nchannels_p = ffi.new('unsigned int *')
    th_sox.libthsox_get_info(str(filepath).encode("utf-8"), bits_per_sample_p, length_p,
                            sample_rate_p, nchannels_p)
    return dict(bits_per_sample=bits_per_sample_p[0],
                nframes=length_p[0] // nchannels_p[0],
                sample_rate=sample_rate_p[0],
                nchannels=nchannels_p[0])


def load(filepath, out=None, offset=0, nframes=None):
    """Read audio from a file into a Tensor

    Args:
      filepath (string): path of audio file to read
      out (Tensor, optional): the result Tensor
      offset (int, optional): offset in frames at which to begin reading.  Must be less than number of frames in file
      nframes (int, optional): number of frames to read.  Fewer frames may be returned if the file is not long enough
    Returns:
      output (Tensor), sample rate (int)

    Notes:
      The use of offsets and nframes does not give exact results when used with mp3 files; these features are best
      used with wavs.
    """
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))
    # initialize output tensor
    if out is not None:
        check_input(out)
    else:
        out = torch.FloatTensor()
    if offset < 0:
        raise TypeError('Expected a non-negative integer, got {}'.format(offset))
    if nframes is None:
        nframes = -1
    elif type(nframes) != int or nframes < 0:
        raise TypeError('Expected None or a non-negative integer, got {}'.format(nframes))

    # load audio signal
    typename = type(out).__name__.replace('Tensor', '')
    func = getattr(th_sox, 'libthsox_{}_read_audio_file'.format(typename))
    sample_rate_p = ffi.new('int*')
    total_frames_p = ffi.new('unsigned long*')
    func(str(filepath).encode("utf-8"), out, sample_rate_p, total_frames_p, offset, nframes)
    if offset > total_frames_p[0]:
        raise IndexError
    sample_rate = sample_rate_p[0]
    # scale to [-1,1]
    out /= 1 << 31
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
