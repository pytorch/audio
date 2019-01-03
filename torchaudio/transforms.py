from __future__ import division, print_function
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

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Scale(object):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".

    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth

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
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)

        return tensor / self.factor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadTrim(object):
    """Pad/Trim a 1d-Tensor (Signal or Labels)

    Args:
        tensor (Tensor): Tensor of audio of size (n x c) or (c x n)
        max_len (int): Length to which the tensor will be padded
        channels_first (bool): Pad for channels first tensors.  Default: `True`

    """

    def __init__(self, max_len, fill_value=0, channels_first=True):
        self.max_len = max_len
        self.fill_value = fill_value
        self.len_dim, self.ch_dim = int(channels_first), int(not channels_first)

    def __call__(self, tensor):
        """

        Returns:
            Tensor: (c x n) or (n x c)

        """
        assert tensor.size(self.ch_dim) < 128, \
            "Too many channels ({}) detected, see channels_first param.".format(tensor.size(self.ch_dim))
        if self.max_len > tensor.size(self.len_dim):
            padding = [self.max_len - tensor.size(self.len_dim)
                       if (i % 2 == 1) and (i // 2 != self.len_dim)
                       else 0
                       for i in range(4)]
            with torch.no_grad():
                tensor = torch.nn.functional.pad(tensor, padding, "constant", self.fill_value)
        elif self.max_len < tensor.size(self.len_dim):
            tensor = tensor.narrow(self.len_dim, 0, self.max_len)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(max_len={0})'.format(self.max_len)


class DownmixMono(object):
    """Downmix any stereo signals to mono.  Consider using a `SoxEffectsChain` with
       the `channels` effect instead of this transformation.

    Inputs:
        tensor (Tensor): Tensor of audio of size (c x n) or (n x c)
        channels_first (bool): Downmix across channels dimension.  Default: `True`

    Returns:
        tensor (Tensor) (Samples x 1):

    """

    def __init__(self, channels_first=None):
        self.ch_dim = int(not channels_first)

    def __call__(self, tensor):
        if not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)

        tensor = torch.mean(tensor, self.ch_dim, True)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class LC2CL(object):
    """Permute a 2d tensor from samples (n x c) to (c x n)
    """

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of audio signal with shape (LxC)

        Returns:
            tensor (Tensor): Tensor of audio signal with shape (CxL)

        """
        return tensor.transpose(0, 1).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPECTROGRAM(object):
    """Create a spectrogram from a raw audio signal

    Args:
        sr (int): sample rate of audio signal
        ws (int): window size, often called the fft size as well
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        n_fft (int, optional): number of fft bins. default: ws // 2 + 1
        pad (int): two sided padding of signal
        window (torch windowing function): default: torch.hann_window
        wkwargs (dict, optional): arguments for window function

    """
    def __init__(self, sr=16000, ws=400, hop=None, n_fft=None,
                 pad=0, window_fn=torch.hann_window, wkwargs=None):
        self.window = window_fn(ws) if wkwargs is None else window_fn(ws, **wkwargs)
        self.sr = sr
        self.ws = ws
        self.hop = hop if hop is not None else ws // 2
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.n_fft = (n_fft - 1) * 2 if n_fft is not None else ws
        self.pad = pad
        self.wkwargs = wkwargs

    def __call__(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (c, n)

        Returns:
            spec_f (Tensor): channels x hops x n_fft (c, l, f), where channels
                is unchanged, hops is the number of hops, and n_fft is the
                number of fourier bins, which should be the window size divided
                by 2 plus 1.

        """
        assert sig.dim() == 2

        if self.pad > 0:
            with torch.no_grad():
                sig = torch.nn.functional.pad(sig, (self.pad, self.pad), "constant")

        spec_f = torch.stft(sig, self.n_fft, self.hop, self.ws,
                            self.window, center=False,
                            normalized=True, onesided=True).transpose(1, 2)
        spec_f /= self.window.pow(2).sum().sqrt()
        spec_f = spec_f.pow(2).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
        return spec_f


class F2M(object):
    """This turns a normal STFT into a MEL Frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.

    Args:
        n_mels (int): number of MEL bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: sr // 2
        f_min (float): minimum frequency. default: 0
    """
    def __init__(self, n_mels=40, sr=16000, f_max=None, f_min=0.):
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min

    def __call__(self, spec_f):

        n_fft = spec_f.size(2)

        m_min = 0. if self.f_min == 0 else 2595 * np.log10(1. + (self.f_min / 700))
        m_max = 2595 * np.log10(1. + (self.f_max / 700))

        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = (700 * (10**(m_pts / 2595) - 1))

        bins = torch.floor(((n_fft - 1) * 2) * f_pts / self.sr).long()

        fb = torch.zeros(n_fft, self.n_mels)
        for m in range(1, self.n_mels + 1):
            f_m_minus = bins[m - 1].item()
            f_m = bins[m].item()
            f_m_plus = bins[m + 1].item()

            if f_m_minus != f_m:
                fb[f_m_minus:f_m, m - 1] = (torch.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
            if f_m != f_m_plus:
                fb[f_m:f_m_plus, m - 1] = (f_m_plus - torch.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

        spec_m = torch.matmul(spec_f, fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m


class SPEC2DB(object):
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.

    Args:
        stype (str): scale of input spectrogram ("power" or "magnitude").  The
            power being the elementwise square of the magnitude. default: "power"
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is -80.
    """
    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        self.top_db = -top_db if top_db > 0 else top_db
        self.multiplier = 10. if stype == "power" else 20.

    def __call__(self, spec):

        spec_db = self.multiplier * torch.log10(spec / spec.max())  # power -> dB
        if self.top_db is not None:
            spec_db = torch.max(spec_db, spec_db.new([self.top_db]))
        return spec_db


class MEL2(object):
    """Create MEL Spectrograms from a raw audio signal using the stft
       function in PyTorch.  Hopefully this solves the speed issue of using
       librosa.

    Sources:
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sr (int): sample rate of audio signal
        ws (int): window size, often called the fft size as well
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        n_fft (int, optional): number of fft bins. default: ws // 2 + 1
        pad (int): two sided padding of signal
        n_mels (int): number of MEL bins
        window (torch windowing function): default: torch.hann_window
        wkwargs (dict, optional): arguments for window function

    Example:
        >>> sig, sr = torchaudio.load("test.wav", normalization=True)
        >>> spec_mel = transforms.MEL2(sr)(sig)  # (c, l, m)
    """
    def __init__(self, sr=16000, ws=400, hop=None, n_fft=None,
                 pad=0, n_mels=40, window_fn=torch.hann_window, wkwargs=None):
        self.window_fn = window_fn
        self.sr = sr
        self.ws = ws
        self.hop = hop if hop is not None else ws // 2
        self.n_fft = n_fft  # number of fourier bins (ws // 2 + 1 by default)
        self.pad = pad
        self.n_mels = n_mels  # number of mel frequency bins
        self.wkwargs = wkwargs
        self.top_db = -80.
        self.f_max = None
        self.f_min = 0.

    def __call__(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])

        Returns:
            spec_mel_db (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.

        """

        transforms = Compose([
            SPECTROGRAM(self.sr, self.ws, self.hop, self.n_fft,
                        self.pad, self.window_fn, self.wkwargs),
            F2M(self.n_mels, self.sr, self.f_max, self.f_min),
            SPEC2DB("power", self.top_db),
        ])

        spec_mel_db = transforms(sig)

        return spec_mel_db


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
            tensor (Tensor): Tensor of audio of size (samples [n] x channels [c])

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
            nparr = tensor[:, i].numpy()  # (samples, )
            sgram = librosa.feature.melspectrogram(
                nparr, **self.kwargs)  # (n_mels, hops)
            L.append(sgram)
        L = np.stack(L, 2)  # (n_mels, hops, channels)
        tensor = torch.from_numpy(L).type_as(tensor)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class BLC2CBL(object):
    """Permute a 3d tensor from Bands x Sample length x Channels to Channels x
       Bands x Samples length
    """

    def __call__(self, tensor):
        """

        Args:
            tensor (Tensor): Tensor of spectrogram with shape (BxLxC)

        Returns:
            tensor (Tensor): Tensor of spectrogram with shape (CxBxL)

        """

        return tensor.permute(2, 0, 1).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MuLawEncoding(object):
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int): Number of channels. default: 256

    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x):
        """

        Args:
            x (FloatTensor/LongTensor or ndarray)

        Returns:
            x_mu (LongTensor or ndarray)

        """
        mu = self.qc - 1.
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
        elif isinstance(x, torch.Tensor):
            if not x.is_floating_point():
                x = x.to(torch.float)
            mu = torch.tensor(mu, dtype=x.dtype)
            x_mu = torch.sign(x) * torch.log1p(mu *
                                               torch.abs(x)) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
        return x_mu

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MuLawExpanding(object):
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int): Number of channels. default: 256

    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x_mu):
        """

        Args:
            x_mu (FloatTensor/LongTensor or ndarray)

        Returns:
            x (FloatTensor or ndarray)

        """
        mu = self.qc - 1.
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, torch.Tensor):
            if not x_mu.is_floating_point():
                x_mu = x_mu.to(torch.float)
            mu = torch.tensor(mu, dtype=x_mu.dtype)
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'
