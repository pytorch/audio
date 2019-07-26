from __future__ import division, print_function
import os.path

import torch
import _torch_sox

import torchaudio


def load(filepath, out=None, normalization=None, num_frames=0, offset=0):
    r"""Loads an audio file from disk into a Tensor.  The default options have
    changed as of torchaudio 0.2 and this function maintains option defaults
    from version 0.1.

    Args:
        filepath (str): Path to audio file
        out (torch.Tensor, optional): An output Tensor to use instead of creating one. (Default: ``None``)
        normalization (bool or number, optional): If boolean `True`, then output is divided by `1 << 31`
            (assumes 16-bit depth audio, and normalizes to `[0, 1]`. If `number`, then output is divided by that
            number. (Default: ``None``)
        num_frames (int, optional): Number of frames to load.  -1 to load everything after the
            offset. (Default: ``0``)
        offset (int, optional): Number of frames from the start of the file to begin data
            loading. (Default: ``0``)

    Returns:
        Tuple[torch.Tensor, int]: The output tensor is of size `[L x C]` where L is the number of audio frames,
        C is the number of channels. The integer is sample-rate of the audio (as listed in the metadata of
        the file)

    Example
        >>> data, sample_rate = torchaudio.legacy.load('foo.mp3')
        >>> print(data.size())
        torch.Size([278756, 2])
        >>> print(sample_rate)
        44100
    """
    return torchaudio.load(filepath, out, normalization, False, num_frames, offset)


def save(filepath, src, sample_rate, precision=32):
    r"""Saves a Tensor with audio signal to disk as a standard format like mp3, wav, etc.
    The default options have changed as of torchaudio 0.2 and this function maintains
    option defaults from version 0.1.

    Args:
        filepath (str): Path to audio file
        src (torch.Tensor): An input 2D Tensor of shape `[L x C]` where L is
            the number of audio frames, C is the number of channels
        sample_rate (int): The sample-rate of the audio to be saved
        precision (int, optional): The bit-precision of the audio to be saved. (Default: ``32``)

    Example
        >>> data, sample_rate = torchaudio.legacy.load('foo.mp3')
        >>> torchaudio.legacy.save('foo.wav', data, sample_rate)
    """
    torchaudio.save(filepath, src, sample_rate, precision, False)
