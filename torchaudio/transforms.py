from __future__ import division
import torch
import numpy as np
try:
    import librosa
except ImportError:
    librosa = None

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

class LC2CL(object):
    """Permute a 2d tensor from samples (Length) x Channels to Channels x
       samples (Length)
    """

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of spectrogram with shape (BxLxC)

        Returns:
            tensor (Tensor): Tensor of spectrogram with shape (CxBxL)

        """

        return tensor.transpose(0, 1).contiguous()


class MEL(object):
    """Create MEL Spectrograms from a raw audio signal. Relatively pretty slow.

       Usage (see librosa.feature.melspectrogram docs):
           MEL(sr=16000, n_fft=1600, hop_length=800, n_mels=64)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio of size (samples x channels)

        Returns:
            tensor (Tensor): n_mels x hops x channels (BxLxC), where n_mels is
                the number of mel bins, hops is the number of hops, and channels
                is unchanged.

        """
        if librosa is None:
            print("librosa not installed, cannot create spectrograms")
            return tensor
        L = []
        for i in range(tensor.size(1)):
            nparr = tensor[:, i].numpy() # (samples, )
            sgram = librosa.feature.melspectrogram(nparr, **self.kwargs) # (n_mels, hops)
            L.append(sgram)
        L = np.stack(L, 2) # (n_mels, hops, channels)
        tensor = torch.from_numpy(L).type_as(tensor)

        return tensor

class BLC2CBL(object):
    """Permute a 3d tensor from Bands x samples (Length) x Channels to Channels x
       Bands x samples (Length)
    """

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of spectrogram with shape (BxLxC)

        Returns:
            tensor (Tensor): Tensor of spectrogram with shape (CxBxL)

        """

        return tensor.permute(2, 0, 1).contiguous()
