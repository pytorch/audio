from __future__ import division, print_function
from warnings import warn
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
        if not tensor.dtype.is_floating_point:
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
        if not tensor.dtype.is_floating_point:
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


def SPECTROGRAM(*args, **kwargs):
    warn("SPECTROGRAM has been renamed to Spectrogram")
    return Spectrogram(*args, **kwargs)


class Spectrogram(object):
    """Create a spectrogram from a raw audio signal

    Args:
        sr (int): sample rate of audio signal
        ws (int): window size
        hop (int, optional): length of hop between STFT windows. default: ws // 2
        n_fft (int, optional): size of fft, creates n_fft // 2 + 1 bins. default: ws
        pad (int): two sided padding of signal
        window (torch windowing function): default: torch.hann_window
        power (int > 0 ) : Exponent for the magnitude spectrogram,
                        e.g., 1 for energy, 2 for power, etc.
        normalize (bool) : whether to normalize by magnitude after stft
        wkwargs (dict, optional): arguments for window function
    """
    def __init__(self, n_fft=400, ws=None, hop=None, 
                 pad=0, window=torch.hann_window,
                 power=2, normalize=False, wkwargs=None):
        self.n_fft = n_fft
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.ws = ws if ws is not None else n_fft
        self.hop = hop if hop is not None else self.ws // 2
        self.window = window(self.ws) if wkwargs is None else window(self.ws, **wkwargs)
        self.pad = pad
        self.power = power
        self.normalize = normalize
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
        self.window = self.window.to(sig.device)

        # default values are consistent with librosa.core.spectrum._spectrogram
        spec_f = torch.stft(sig, self.n_fft, self.hop, self.ws,
                            self.window, center=True,
                            normalized=False, onesided=True,
                            pad_mode='reflect').transpose(1, 2)
        if self.normalize: 
            spec_f /= self.window.pow(2).sum().sqrt()
        spec_f = spec_f.pow(self.power).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
        return spec_f


def F2M(*args, **kwargs):
    warn("F2M has been renamed to MelScale")
    return MelScale(*args, **kwargs)


class MelScale(object):
    """This turns a normal STFT into a mel frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.

    Args:
        n_mels (int): number of mel bins
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: `sr` // 2
        f_min (float): minimum frequency. default: 0
        n_stft (int, optional): number of filter banks from stft. Calculated from first input
            if `None` is given.  See `n_fft` in `Spectrogram`.
    """
    def __init__(self, n_mels=128, sr=16000, f_max=None, f_min=0., n_stft=None):
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.fb = self._create_fb_matrix(n_stft) if n_stft is not None else n_stft

    def __call__(self, spec_f):
        if self.fb is None:
            self.fb = self._create_fb_matrix(spec_f.size(2)).to(spec_f.device)
        else:
            # need to ensure same device for dot product
            self.fb.to(spec_f.device)
        spec_m = torch.matmul(spec_f, self.fb)  # (c, l, n_fft) dot (n_fft, n_mels) -> (c, l, n_mels)
        return spec_m

    def _create_fb_matrix(self, n_stft):
        """ Create a frequency bin conversion matrix.

        Args:
            n_stft (int): number of filter banks from spectrogram
        """

        # get stft freq bins
        stft_freqs = torch.linspace(self.f_min, self.f_max, n_stft)
        # calculate mel freq bins
        m_min = 0. if self.f_min == 0 else self._hertz_to_mel(self.f_min)
        m_max = self._hertz_to_mel(self.f_max)
        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_pts = self._mel_to_hertz(m_pts)
        # calculate the difference between each mel point and each stft freq point in hertz
        f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
        slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)  # (n_stft, n_mels + 2)
        # create overlapping triangles
        z = torch.tensor(0.)
        down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
        fb = torch.max(z, torch.min(down_slopes, up_slopes))
        return fb

    def _hertz_to_mel(self, f):
        return 2595. * torch.log10(torch.tensor(1.) + (f / 700.))

    def _mel_to_hertz(self, mel):
        return 700. * (10**(mel / 2595.) - 1.)




class SpectogramToDB(object):
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.

    Args:
        stype (str): scale of input spectrogram ("power" or "magnitude").  The
            power being the elementwise square of the magnitude. default: "power"
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is -80.
    """
    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        self.top_db = top_db
        self.multiplier = 10. if stype == "power" else 20.

        self.amin = torch.tensor([1e-10])
        self.ref_value = torch.tensor([1.0])

    def __call__(self, spec):

        # numerically stable implementation from librosa
        spec_db = self.multiplier * torch.log10(torch.max(self.amin.to(spec.device), spec))
        spec_db -= self.multiplier * torch.log10(torch.max(self.amin.to(spec.device), self.ref_value))

        if self.top_db is not None:
            spec_db = torch.max(spec_db, spec_db.new_full((1,), spec_db.max() - self.top_db))
        return spec_db


class MFCC(object):
    """Create the Mel-frequency cepstrum coefficients from an audio signal

        Args:
        sr (int) : sample rate of audio signal
        n_mfcc (int) : number of mfc coefficients to retain
        dct_type (int) : type of DCT (discrete cosine transform) to use
        norm (string) : norm to use
        melkwargs (dict, optional): arguments for MelSpectrogram
    """
    def __init__(self, sr=16000, n_mfcc=40, dct_type=2, norm='ortho', melkwargs=None):

        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported'.format(dct_type))
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.melkwargs = melkwargs
        self.top_db = 80.
        self.s2db = SpectrogramToDB("power", self.top_db)
        self.MelSpectrogram = MelSpectrogram(sr=self.sr, **melkwargs) if melkwargs is not None else MelSpectrogram(sr=self.sr)

    def __call__(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])

        Returns:
            spec_mel_db (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.

        """
        mel_spect_db = self.s2db(self.MelSpectrogram(sig))
        mfcc = dct(mel_spect_db, norm=self.norm)[:, :, :self.n_mfcc]
        return mfcc


class MelSpectrogram(object):
    """Create MEL Spectrograms from a raw audio signal using the stft
       function in PyTorch.

    Sources:
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sr (int): sample rate of audio signal
        ws (int): window size
        hop (int, optional): length of hop between STFT windows. default: `ws` // 2
        n_fft (int, optional): number of fft bins. default: `ws` // 2 + 1
        f_max (float, optional): maximum frequency. default: `sr` // 2
        f_min (float): minimum frequency. default: 0
        pad (int): two sided padding of signal
        n_mels (int): number of MEL bins
        window (torch windowing function): default: `torch.hann_window`
        wkwargs (dict, optional): arguments for window function

    Example:
        >>> sig, sr = torchaudio.load("test.wav", normalization=True)
        >>> spec_mel = transforms.MelSpectrogram(sr)(sig)  # (c, l, m)
    """
    def __init__(self, sr=16000, n_fft=400, ws=None, hop=None, f_min=0., f_max=None,
                 pad=0, n_mels=128, window=torch.hann_window, wkwargs=None):
        self.window = window
        self.sr = sr
        self.n_fft = n_fft
        self.ws = ws if ws is not None else n_fft
        self.hop = hop if hop is not None else self.ws // 2
        self.pad = pad
        self.n_mels = n_mels  # number of mel frequency bins
        self.wkwargs = wkwargs
        self.f_max = f_max
        self.f_min = f_min
        self.spec = Spectrogram(n_fft=self.n_fft, ws=self.ws, hop=self.hop,
                                pad=self.pad, window=self.window, power=2,
                                normalize=False, wkwargs=self.wkwargs)
        self.fm = MelScale(self.n_mels, self.sr, self.f_max, self.f_min)
        self.transforms = Compose([
            self.spec, self.fm
        ])

    def __call__(self, sig):
        """
        Args: 
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])

        Returns:
            spec_mel_db (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.

        """
        spec_mel_db = self.transforms(sig)

        return spec_mel_db


def MEL(*args, **kwargs):
    raise DeprecationWarning("MEL has been removed from the library please use MelSpectrogram or librosa")


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
            if not x.dtype.is_floating_point:
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
            if not x_mu.dtype.is_floating_point:
                x_mu = x_mu.to(torch.float)
            mu = torch.tensor(mu, dtype=x_mu.dtype)
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x

    def __repr__(self):
        return self.__class__.__name__ + '()'


def dct(x, norm=None):
    """
    from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V
