from __future__ import division
import torch
import numpy as np

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Scale(),
        >>>     transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

class Scale(object):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".

    Args:
        factor (float): maximum value of input tensor. default: 16-bit depth

    """

    def __init__(self, factor=2**31):
        self.factor = factor

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)

        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)

        """
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()

        return tensor / self.factor

class PadTrim(object):
    """Pad/Trim a 1d-Tensor (Signal or Labels)

    """

    def __init__(self, max_len, fill_value=0):
        self.max_len = max_len
        self.fill_value = fill_value

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)
            max_len (int): Length to which the tensor will be padded

        Returns:
            Tensor: (max_len x Channels)

        """
        if self.max_len > tensor.size(0):
            pad = torch.ones((self.max_len-tensor.size(0),
                              tensor.size(1))) * self.fill_value
            pad = pad.type_as(tensor)
            tensor = torch.cat((tensor, pad), dim=0)
        elif self.max_len < tensor.size(0):
            tensor = tensor[:self.max_len, :]
        return tensor


class DownmixMono(object):
    """Downmix any stereo signals to mono

    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)

        Returns:
            Tensor: (Samples x 1)

        """
        if isinstance(tensor, (torch.LongTensor, torch.IntTensor)):
            tensor = tensor.float()

        if tensor.size(1) > 1:
            tensor = torch.mean(tensor.float(), 1, True)
        return tensor
