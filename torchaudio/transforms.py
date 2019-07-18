from __future__ import division, print_function
from warnings import warn
import math
import torch
from typing import Optional
from . import functional as F
from .compliance import kaldi


class Scale(torch.jit.ScriptModule):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".

    Args:
        factor (float): maximum value of input tensor. default: 16-bit depth
    """
    __constants__ = ['factor']

    def __init__(self, factor=float(2**31)):
        super(Scale, self).__init__()
        self.factor = factor

    @torch.jit.script_method
    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio of size (c, n)

        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)
        """
        return F.scale(tensor, self.factor)


class PadTrim(torch.jit.ScriptModule):
    """Pad/Trim a 2D-Tensor (Signal or Labels)

    Args:
        tensor (Tensor): Tensor of audio of size (c, n)
        max_len (int): Length to which the tensor will be padded
    """
    __constants__ = ['max_len', 'fill_value']

    def __init__(self, max_len, fill_value=0.):
        super(PadTrim, self).__init__()
        self.max_len = max_len
        self.fill_value = fill_value

    @torch.jit.script_method
    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio of size (c, n)

        Returns:
            Tensor: (c, `max_len`)
        """
        return F.pad_trim(tensor, self.max_len, self.fill_value)


class DownmixMono(torch.jit.ScriptModule):
    """Downmix any stereo signals to mono.  Consider using a `SoxEffectsChain` with
       the `channels` effect instead of this transformation.

    Inputs:
        tensor (Tensor): Tensor of audio of size (c, n)

    Returns:
        tensor (Tensor) (1, n):
    """

    def __init__(self):
        super(DownmixMono, self).__init__()

    @torch.jit.script_method
    def forward(self, tensor):
        return F.downmix_mono(tensor, self.ch_dim)


class LC2CL(torch.jit.ScriptModule):
    """Converts a 2D tensor from (n, c) to (c, n)
    """
    def __init__(self):
        super(LC2CL, self).__init__()

    @torch.jit.script_method
    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio signal with shape (n, c)
        Returns:
            tensor (Tensor): Tensor of audio signal with shape (c, n)
        """
        return F.LC2CL(tensor)


class Spectrogram(torch.jit.ScriptModule):
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
    __constants__ = ['n_fft', 'ws', 'hop', 'pad', 'power', 'normalize']

    def __init__(self, n_fft=400, ws=None, hop=None,
                 pad=0, window=torch.hann_window,
                 power=2, normalize=False, wkwargs=None):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of fft bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.ws = ws if ws is not None else n_fft
        self.hop = hop if hop is not None else self.ws // 2
        window = window(self.ws) if wkwargs is None else window(self.ws, **wkwargs)
        self.window = torch.jit.Attribute(window, torch.Tensor)
        self.pad = pad
        self.power = power
        self.normalize = normalize

    @torch.jit.script_method
    def forward(self, sig):
        """
        Args:
            sig (Tensor): Tensor of audio of size (c, n)

        Returns:
            spec_f (Tensor): Channels x frequency x time (c, f, t), where channels
            is unchanged, frequency is `n_fft // 2 + 1` where `n_fft` is the number of
            fourier bins, time is the number of window hops

        """
        return F.spectrogram(sig, self.pad, self.window, self.n_fft, self.hop,
                             self.ws, self.power, self.normalize)


class SpectrogramToDB(torch.jit.ScriptModule):
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
    __constants__ = ['multiplier', 'amin', 'ref_value', 'db_multiplier']

    def __init__(self, stype="power", top_db=None):
        super(SpectrogramToDB, self).__init__()
        self.stype = torch.jit.Attribute(stype, str)
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = torch.jit.Attribute(top_db, Optional[float])
        self.multiplier = 10. if stype == "power" else 20.
        self.amin = 1e-10
        self.ref_value = 1.
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    @torch.jit.script_method
    def forward(self, spec):
        r"""Numerically stable implementation from Librosa
        https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html

        Args:
            spec (torch.Tensor): STFT of size (c, f, t)

        Returns:
            torch.Tensor: STFT after changing scale of size (c, f, t)
        """
        return F.spectrogram_to_DB(spec, self.multiplier, self.amin, self.db_multiplier, self.top_db)


class MelScale(torch.jit.ScriptModule):
    """This turns a normal STFT into a mel frequency STFT, using a conversion
       matrix.  This uses triangular filter banks.

       User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int): Number of mel filterbanks
        sr (int): sample rate of audio signal
        f_max (float, optional): maximum frequency. default: `sr` // 2
        f_min (float): minimum frequency. default: 0
        n_stft (int, optional): number of filter banks from stft. Calculated from first input
            if `None` is given.  See `n_fft` in `Spectrogram`.
    """
    __constants__ = ['n_mels', 'sr', 'f_min', 'f_max']

    def __init__(self, n_mels=128, sr=16000, f_max=None, f_min=0., n_stft=None):
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.f_max = f_max if f_max is not None else float(sr // 2)
        self.f_min = f_min
        fb = torch.empty(0) if n_stft is None else F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels)
        self.fb = torch.jit.Attribute(fb, torch.Tensor)

    @torch.jit.script_method
    def forward(self, spec_f):
        r"""
        Args:
            spec_f (torch.Tensor): a spectrogram STFT of size (c, f, t)

        Returns:
            torch.Tensor: mel frequency spectrogram of size (c, `n_mels`, t)
        """
        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(spec_f.size(1), self.f_min, self.f_max, self.n_mels)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (c, f, t).transpose(...) dot (f, n_mels) -> (c, t, n_mels).transpose(...)
        spec_m = torch.matmul(spec_f.transpose(1, 2), self.fb).transpose(1, 2)
        return spec_m


class MelSpectrogram(torch.jit.ScriptModule):
    """Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram
    and MelScale.

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
        n_mels (int): Number of mel filterbanks
        window (torch windowing function): default: `torch.hann_window`
        wkwargs (dict, optional): arguments for window function

    Example:
        >>> sig, sr = torchaudio.load("test.wav", normalization=True)
        >>> spec_mel = transforms.MelSpectrogram(sr)(sig)  # (c, n_mels, t)
    """
    __constants__ = ['sr', 'n_fft', 'ws', 'hop', 'pad', 'n_mels', 'f_min']

    def __init__(self, sr=16000, n_fft=400, ws=None, hop=None, f_min=0., f_max=None,
                 pad=0, n_mels=128, window=torch.hann_window, wkwargs=None):
        super(MelSpectrogram, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.ws = ws if ws is not None else n_fft
        self.hop = hop if hop is not None else self.ws // 2
        self.pad = pad
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = torch.jit.Attribute(f_max, Optional[float])
        self.f_min = f_min
        self.spec = Spectrogram(n_fft=self.n_fft, ws=self.ws, hop=self.hop,
                                pad=self.pad, window=window, power=2,
                                normalize=False, wkwargs=wkwargs)
        self.fm = MelScale(self.n_mels, self.sr, self.f_max, self.f_min)

    @torch.jit.script_method
    def forward(self, sig):
        """
        Args:
            sig (torch.Tensor): Tensor of audio of size (c, n)

        Returns:
            torch.Tensor: mel frequency spectrogram of size (c, `n_mels`, t)
        """
        spec = self.spec(sig)
        spec_mel = self.fm(spec)
        return spec_mel


class MFCC(torch.jit.ScriptModule):
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
        norm (string, optional) : norm to use
        log_mels (bool) : whether to use log-mel spectrograms instead of db-scaled
        melkwargs (dict, optional): arguments for MelSpectrogram
    """
    __constants__ = ['sr', 'n_mfcc', 'dct_type', 'top_db', 'log_mels']

    def __init__(self, sr=16000, n_mfcc=40, dct_type=2, norm='ortho', log_mels=False,
                 melkwargs=None):
        super(MFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported'.format(dct_type))
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = torch.jit.Attribute(norm, Optional[str])
        self.top_db = 80.
        self.s2db = SpectrogramToDB("power", self.top_db)

        if melkwargs is not None:
            self.MelSpectrogram = MelSpectrogram(sr=self.sr, **melkwargs)
        else:
            self.MelSpectrogram = MelSpectrogram(sr=self.sr)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.dct_mat = torch.jit.Attribute(dct_mat, torch.Tensor)
        self.log_mels = log_mels

    @torch.jit.script_method
    def forward(self, sig):
        """
        Args:
            sig (torch.Tensor): Tensor of audio of size (c, n)

        Returns:
            torch.Tensor: spec_mel_db of size (c, `n_mfcc`, t)
        """
        mel_spect = self.MelSpectrogram(sig)
        if self.log_mels:
            log_offset = 1e-6
            mel_spect = torch.log(mel_spect + log_offset)
        else:
            mel_spect = self.s2db(mel_spect)
        # (c, `n_mels`, t).tranpose(...) dot (`n_mels`, `n_mfcc`) -> (c, t, `n_mfcc`).tranpose(...)
        mfcc = torch.matmul(mel_spect.transpose(1, 2), self.dct_mat).transpose(1, 2)
        return mfcc


class MuLawEncoding(torch.jit.ScriptModule):
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int): Number of channels. default: 256

    """
    __constants__ = ['qc']

    def __init__(self, quantization_channels=256):
        super(MuLawEncoding, self).__init__()
        self.qc = quantization_channels

    @torch.jit.script_method
    def forward(self, x):
        """
        Args:
            x (FloatTensor/LongTensor)

        Returns:
            x_mu (LongTensor)
        """
        return F.mu_law_encoding(x, self.qc)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MuLawExpanding(torch.jit.ScriptModule):
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int): Number of channels. default: 256

    """
    __constants__ = ['qc']

    def __init__(self, quantization_channels=256):
        super(MuLawExpanding, self).__init__()
        self.qc = quantization_channels

    @torch.jit.script_method
    def forward(self, x_mu):
        """
        Args:
            x_mu (Tensor)

        Returns:
            x (Tensor)
        """
        return F.mu_law_expanding(x_mu, self.qc)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resample(torch.nn.Module):
    """Resamples a signal from one frequency to another. A resampling method can
    be given.

    Args:
        orig_freq (float): the original frequency of the signal
        new_freq (float): the desired frequency
        resampling_method (str): the resampling method (Default: 'kaldi' which uses
            sinc interpolation)
    """
    def __init__(self, orig_freq, new_freq, resampling_method='sinc_interpolation'):
        super(Resample, self).__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampling_method = resampling_method

    def forward(self, sig):
        """
        Args:
            sig (Tensor): the input signal of size (c, n)

        Returns:
            Tensor: output signal of size (c, m)
        """
        if self.resampling_method == 'sinc_interpolation':
            return kaldi.resample_waveform(sig, self.orig_freq, self.new_freq)

        raise ValueError('Invalid resampling method: %s' % (self.resampling_method))
