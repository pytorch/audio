from __future__ import division, print_function
import os.path

import torch
import _torch_sox

import torchaudio


def load(filepath, out=None, normalization=None, num_frames=0, offset=0):
    """Loads an audio file from disk into a Tensor.  The default options have
    changed as of torchaudio 0.2 and this function maintains option defaults
    from version 0.1.

    Args:
        filepath (string): path to audio file
        out (Tensor, optional): an output Tensor to use instead of creating one
        normalization (bool or number, optional): If boolean `True`, then output is divided by `1 << 31`
                                                  (assumes 16-bit depth audio, and normalizes to `[0, 1]`.
                                                  If `number`, then output is divided by that number
        num_frames (int, optional): number of frames to load.  -1 to load everything after the offset.
        offset (int, optional): number of frames from the start of the file to begin data loading.

    Returns: tuple(Tensor, int)
       - Tensor: output Tensor of size `[L x C]` where L is the number of audio frames, C is the number of channels
       - int: the sample-rate of the audio (as listed in the metadata of the file)

    Example::

        >>> data, sample_rate = torchaudio.legacy.load('foo.mp3')
        >>> print(data.size())
        torch.Size([278756, 2])
        >>> print(sample_rate)
        44100

    """
    return torchaudio.load(filepath, out, normalization, False, num_frames, offset)


def save(filepath, src, sample_rate, precision=32):
    """Saves a Tensor with audio signal to disk as a standard format like mp3, wav, etc.
    The default options have changed as of torchaudio 0.2 and this function maintains
    option defaults from version 0.1.

    Args:
        filepath (string): path to audio file
        src (Tensor): an input 2D Tensor of shape `[L x C]` where L is
                      the number of audio frames, C is the number of channels
        sample_rate (int): the sample-rate of the audio to be saved
        precision (int, optional): the bit-precision of the audio to be saved

    Example::

        >>> data, sample_rate = torchaudio.legacy.load('foo.mp3')
        >>> torchaudio.legacy.save('foo.wav', data, sample_rate)

    """
    torchaudio.save(filepath, src, sample_rate, precision, False)
