import os.path

import torch
import _torch_sox

from torchaudio import transforms
from torchaudio import datasets


def get_tensor_type_name(tensor):
    return tensor.type().replace('torch.', '').replace('Tensor', '')


def check_input(src):
    if not torch.is_tensor(src):
        raise TypeError('Expected a tensor, got %s' % type(src))
    if src.is_cuda:
        raise TypeError('Expected a CPU based tensor, got %s' % type(src))


def load(filepath, out=None, normalization=None):
    """Loads an audio file from disk into a Tensor

    Args:
        filepath (string): path to audio file
        out (Tensor, optional): an output Tensor to use instead of creating one
        normalization (bool or number, optional): If boolean `True`, then output is divided by `1 << 31`
                                                  (assumes 16-bit depth audio, and normalizes to `[0, 1]`.
                                                  If `number`, then output is divided by that number

    Returns: tuple(Tensor, int)
       - Tensor: output Tensor of size `[L x C]` where L is the number of audio frames, C is the number of channels
       - int: the sample-rate of the audio (as listed in the metadata of the file)

    Example::

        >>> data, sample_rate = torchaudio.load('foo.mp3')
        >>> print(data.size())
        torch.Size([278756, 2])
        >>> print(sample_rate)
        44100

    """
    # check if valid file
    if not os.path.isfile(filepath):
        raise OSError("{} not found or is a directory".format(filepath))

    # initialize output tensor
    if out is not None:
        check_input(out)
    else:
        out = torch.FloatTensor()

    sample_rate = _torch_sox.read_audio_file(filepath, out)
    # normalize if needed
    if isinstance(normalization, bool) and normalization:
        out /= 1 << 31  # assuming 16-bit depth
    elif isinstance(normalization, (float, int)):
        out /= normalization  # normalize with custom value

    return out, sample_rate


def save(filepath, src, sample_rate):
    """Saves a Tensor with audio signal to disk as a standard format like mp3, wav, etc.

    Args:
        filepath (string): path to audio file
        src (Tensor): an input 2D Tensor of shape `[L x C]` where L is
                      the number of audio frames, C is the number of channels
        sample_rate (int): the sample-rate of the audio to be saved

    Example::

        >>> data, sample_rate = torchaudio.load('foo.mp3')
        >>> torchaudio.save('foo.wav', data, sample_rate)

    """
    # check if save directory exists
    abs_dirpath = os.path.dirname(os.path.abspath(filepath))
    if not os.path.isdir(abs_dirpath):
        raise OSError("Directory does not exist: {}".format(abs_dirpath))
    # Check/Fix shape of source data
    if len(src.size()) == 1:
        # 1d tensors as assumed to be mono signals
        src.unsqueeze_(1)
    elif len(src.size()) > 2 or src.size(1) > 2:
        raise ValueError(
            "Expected format (L x N), N = 1 or 2, but found {}".format(src.size()))
    # check if sample_rate is an integer
    if not isinstance(sample_rate, int):
        if int(sample_rate) == sample_rate:
            sample_rate = int(sample_rate)
        else:
            raise TypeError('Sample rate should be a integer')
    # programs such as librosa normalize the signal, unnormalize if detected
    if src.min() >= -1.0 and src.max() <= 1.0:
        src = src * (1 << 31)  # assuming 16-bit depth
        src = src.long()
    # save data to file
    extension = os.path.splitext(filepath)[1]
    check_input(src)
    _torch_sox.write_audio_file(filepath, src, extension[1:], sample_rate)
