from __future__ import absolute_import, division, print_function, unicode_literals
from warnings import warn
import math
import torch
from . import functional as F


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
        return F.scale(tensor, self.factor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class PadTrim(object):
    """Pad/Trim a 2d-Tensor (Signal or Labels)

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
        return F.pad_trim(tensor, self.ch_dim, self.max_len, self.len_dim, self.fill_value)

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
        return F.downmix_mono(tensor, self.ch_dim)

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
        return F.LC2CL(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def SPECTROGRAM(*args, **kwargs):
    warn("SPECTROGRAM has been renamed to Spectrogram")
    return Spectrogram(*args, **kwargs)


class Spectrogram(object):
    """Create a spectrogram from a raw audio signal

    Args:
        n_fft (int, optional): size of fft, creates n_fft // 2 + 1 bins
        ws (int): window size. default: n_fft
        hop (int, optional): length of hop between STFT windows. default: ws // 2
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
        return F.spectrogram(sig, self.pad, self.window, self.n_fft, self.hop,
                             self.ws, self.power, self.normalize)


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
        self.fb = F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels) if n_stft is not None else n_stft

    def __call__(self, spec_f):
        self.fb, spec_m = F.mel_scale(spec_f, self.f_min, self.f_max, self.n_mels, self.fb)
        return spec_m


class SpectrogramToDB(object):
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        stype (str): scale of input spectrogram ("power" or "magnitude").  The
            power being the elementwise square of the magnitude. default: "power"
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is 80.
    """
    def __init__(self, stype="power", top_db=None):
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10. if stype == "power" else 20.
        self.amin = 1e-10
        self.ref_value = 1.
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def __call__(self, spec):
        # numerically stable implementation from librosa
        # https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html
        return F.spectrogram_to_DB(spec, self.multiplier, self.amin, self.db_multiplier, self.top_db)


class MFCC(object):
    """Create the Mel-frequency cepstrum coefficients from an audio signal

        By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
        This is not the textbook implementation, but is implemented here to
        give consistency with librosa.

        This output depends on the maximum value in the input spectrogram, and so
        may return different values for an audio clip split into snippets vs. a
        a full clip.

        Args:
        sr (int) : sample rate of audio signal
        n_mfcc (int) : number of mfc coefficients to retain
        dct_type (int) : type of DCT (discrete cosine transform) to use
        norm (string) : norm to use
        log_mels (bool) : whether to use log-mel spectrograms instead of db-scaled
        melkwargs (dict, optional): arguments for MelSpectrogram
    """
    def __init__(self, sr=16000, n_mfcc=40, dct_type=2, norm='ortho', log_mels=False,
                 melkwargs=None):

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

        if melkwargs is not None:
            self.MelSpectrogram = MelSpectrogram(sr=self.sr, **melkwargs)
        else:
            self.MelSpectrogram = MelSpectrogram(sr=self.sr)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        self.dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.log_mels = log_mels

    def __call__(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (channels [c], samples [n])

        Returns:
            spec_mel_db (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.
        """
        return F.MFCC(sig, self.MelSpectrogram(sig), self.log_mels, self.s2db, self.dct_mat)


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
            spec_mel (Tensor): channels x hops x n_mels (c, l, m), where channels
                is unchanged, hops is the number of hops, and n_mels is the
                number of mel bins.

        """
        spec_mel = self.transforms(sig)

        return spec_mel


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
        return F.BLC2CBL(tensor)

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
            x (FloatTensor/LongTensor)

        Returns:
            x_mu (LongTensor)

        """
        return F.mu_law_encoding(x, self.qc)

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
            x_mu (Tensor)

        Returns:
            x (Tensor)

        """
        return F.mu_law_expanding(x_mu, self.qc)

    def __repr__(self):
        return self.__class__.__name__ + '()'
