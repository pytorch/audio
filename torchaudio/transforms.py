from __future__ import absolute_import, division, print_function, unicode_literals
from warnings import warn
import math
import torch
from typing import Optional
from . import functional as F
from .compliance import kaldi


__all__ = [
    'Spectrogram',
    'GriffinLim',
    'AmplitudeToDB',
    'MelScale',
    'MelSpectrogram',
    'MFCC',
    'MuLawEncoding',
    'MuLawDecoding',
    'Resample',
    'ComplexNorm',
    'TimeStretch',
    'FrequencyMasking',
    'TimeMasking',
]


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins
        win_length (int): Window size. (Default: ``n_fft``)
        hop_length (int, optional): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        pad (int): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[[...], torch.Tensor]): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (int): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...]): Arguments for window function. (Default: ``None``)
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self, n_fft=400, win_length=None, hop_length=None,
                 pad=0, window_fn=torch.hann_window,
                 power=2, normalized=False, wkwargs=None):
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequecies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (channel, time)

        Returns:
            torch.Tensor: Dimension (channel, freq, time), where channel
            is unchanged, freq is ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return F.spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length,
                             self.win_length, self.power, self.normalized)


class GriffinLim(torch.nn.Module):
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
        Implementation ported from `librosa`.

    .. [1] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto.
        "librosa: Audio and music signal analysis in python."
        In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

    .. [2] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.

    .. [3] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins
        n_iter (int, optional): Number of iteration for phase recovery process.
        win_length (int): Window size. (Default: ``n_fft``)
        hop_length (int, optional): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        window_fn (Callable[[...], torch.Tensor]): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (int): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...]): Arguments for window function. (Default: ``None``)
        momentum (float): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge. (Default: 0.99)
        length (int, optional): Array length of the expected output. (Default: ``None``)
        rand_init (bool): Initializes phase randomly if True and to zero otherwise. (Default: ``True``)
    """
    __constants__ = ['n_fft', 'n_iter', 'win_length', 'hop_length', 'power', 'normalized',
                     'length', 'momentum', 'rand_init']

    def __init__(self, n_fft=400, n_iter=32, hop_length=None, win_length=None,
                 window_fn=torch.hann_window, wkwargs=None, normalized=False,
                 power=2, length=None, momentum=0.99, rand_init=True):
        super(GriffinLim, self).__init__()

        assert momentum < 1, 'momentum=%s > 1 can be unstable' % momentum
        assert momentum > 0, 'momentum=%s < 0' % momentum

        self.n_fft = n_fft
        self.n_iter = n_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.normalized = normalized
        self.length = length
        self.power = power
        self.momentum = momentum / (1 + momentum)
        self.rand_init = rand_init

    def forward(self, S):
        return F.griffinlim(S, self.window, self.n_fft, self.hop_length, self.win_length, self.power,
                            self.normalized, self.n_iter, self.momentum, self.length, self.rand_init)


class AmplitudeToDB(torch.jit.ScriptModule):
    r"""Turn a tensor from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        stype (str): scale of input tensor ('power' or 'magnitude'). The
            power being the elementwise square of the magnitude. (Default: ``'power'``)
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is 80. (Default: ``None``)
    """
    __constants__ = ['multiplier', 'amin', 'ref_value', 'db_multiplier']

    def __init__(self, stype='power', top_db=None):
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = torch.jit.Attribute(top_db, Optional[float])
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x):
        r"""Numerically stable implementation from Librosa
        https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html

        Args:
            x (torch.Tensor): Input tensor before being converted to decibel scale

        Returns:
            torch.Tensor: Output tensor in decibel scale
        """
        return F.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)


class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int): Sample rate of audio signal. (Default: ``16000``)
        f_min (float): Minimum frequency. (Default: ``0.``)
        f_max (float, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See ``n_fft`` in :class:`Spectrogram`.
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self, n_mels=128, sample_rate=16000, f_min=0., f_max=None, n_stft=None):
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)

        fb = torch.empty(0) if n_stft is None else F.create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate)
        self.register_buffer('fb', fb)

    def forward(self, specgram):
        r"""
        Args:
            specgram (torch.Tensor): A spectrogram STFT of dimension (channel, freq, time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (channel, ``n_mels``, time)
        """
        if self.fb.numel() == 0:
            tmp_fb = F.create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)
        return mel_specgram


class MelSpectrogram(torch.nn.Module):
    r"""Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram
    and MelScale.

    Sources
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: ``16000``)
        win_length (int): Window size. (Default: ``n_fft``)
        hop_length (int, optional): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins
        f_min (float): Minimum frequency. (Default: ``0.``)
        f_max (float, optional): Maximum frequency. (Default: ``None``)
        pad (int): Two sided padding of signal. (Default: ``0``)
        n_mels (int): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[[...], torch.Tensor]): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        wkwargs (Dict[..., ...]): Arguments for window function. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalization=True)
        >>> mel_specgram = transforms.MelSpectrogram(sample_rate)(waveform)  # (channel, n_mels, time)
    """
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min']

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None, f_min=0., f_max=None,
                 pad=0, n_mels=128, window_fn=torch.hann_window, wkwargs=None):
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=2,
                                       normalized=False, wkwargs=wkwargs)
        self.mel_scale = MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1)

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (channel, time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (channel, ``n_mels``, time)
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class MFCC(torch.nn.Module):
    r"""Create the Mel-frequency cepstrum coefficients from an audio signal

    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int): Sample rate of audio signal. (Default: ``16000``)
        n_mfcc (int): Number of mfc coefficients to retain. (Default: ``40``)
        dct_type (int): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``'ortho'``)
        log_mels (bool): whether to use log-mel spectrograms instead of db-scaled. (Default:
            ``False``)
        melkwargs (dict, optional): arguments for MelSpectrogram. (Default: ``None``)
    """
    __constants__ = ['sample_rate', 'n_mfcc', 'dct_type', 'top_db', 'log_mels']

    def __init__(self, sample_rate=16000, n_mfcc=40, dct_type=2, norm='ortho', log_mels=False,
                 melkwargs=None):
        super(MFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported'.format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)

        if melkwargs is not None:
            self.MelSpectrogram = MelSpectrogram(sample_rate=self.sample_rate, **melkwargs)
        else:
            self.MelSpectrogram = MelSpectrogram(sample_rate=self.sample_rate)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_mels = log_mels

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (channel, time)

        Returns:
            torch.Tensor: specgram_mel_db of size (channel, ``n_mfcc``, time)
        """
        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = torch.matmul(mel_specgram.transpose(1, 2), self.dct_mat).transpose(1, 2)
        return mfcc


class MuLawEncoding(torch.nn.Module):
    r"""Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int): Number of channels (Default: ``256``)
    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels=256):
        super(MuLawEncoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x):
        r"""
        Args:
            x (torch.Tensor): A signal to be encoded

        Returns:
            x_mu (torch.Tensor): An encoded signal
        """
        return F.mu_law_encoding(x, self.quantization_channels)


class MuLawDecoding(torch.nn.Module):
    r"""Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int): Number of channels (Default: ``256``)
    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels=256):
        super(MuLawDecoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x_mu):
        r"""
        Args:
            x_mu (torch.Tensor): A mu-law encoded signal which needs to be decoded

        Returns:
            torch.Tensor: The signal decoded
        """
        return F.mu_law_decoding(x_mu, self.quantization_channels)


class Resample(torch.nn.Module):
    r"""Resample a signal from one frequency to another. A resampling method can
    be given.

    Args:
        orig_freq (float): The original frequency of the signal. (Default: ``16000``)
        new_freq (float): The desired frequency. (Default: ``16000``)
        resampling_method (str): The resampling method (Default: ``'sinc_interpolation'``)
    """
    def __init__(self, orig_freq=16000, new_freq=16000, resampling_method='sinc_interpolation'):
        super(Resample, self).__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampling_method = resampling_method

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): The input signal of dimension (channel, time)

        Returns:
            torch.Tensor: Output signal of dimension (channel, time)
        """
        if self.resampling_method == 'sinc_interpolation':
            return kaldi.resample_waveform(waveform, self.orig_freq, self.new_freq)

        raise ValueError('Invalid resampling method: %s' % (self.resampling_method))


class ComplexNorm(torch.nn.Module):
    r"""Compute the norm of complex tensor input
    Args:
        power (float): Power of the norm. Defaults to `1.0`.
    """
    __constants__ = ['power']

    def __init__(self, power=1.0):
        super(ComplexNorm, self).__init__()
        self.power = power

    def forward(self, complex_tensor):
        r"""
        Args:
            complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        Returns:
            Tensor: norm of the input tensor, shape of `(..., )`
        """
        return F.complex_norm(complex_tensor, self.power)


class ComputeDeltas(torch.nn.Module):
    r"""Compute delta coefficients of a tensor, usually a spectrogram.

    See `torchaudio.functional.compute_deltas` for more details.

    Args:
        win_length (int): The window length used for computing delta.
    """
    __constants__ = ['win_length']

    def __init__(self, win_length=5, mode="replicate"):
        super(ComputeDeltas, self).__init__()
        self.win_length = win_length
        self.mode = mode

    def forward(self, specgram):
        r"""
        Args:
            specgram (torch.Tensor): Tensor of audio of dimension (channel, freq, time)

        Returns:
            deltas (torch.Tensor): Tensor of audio of dimension (channel, freq, time)
        """
        return F.compute_deltas(specgram, win_length=self.win_length, mode=self.mode)


class TimeStretch(torch.jit.ScriptModule):
    r"""Stretch stft in time without modifying pitch for a given rate.

    Args:
        hop_length (int): Number audio of frames between STFT columns. (Default: ``n_fft // 2``)
        n_freq (int, optional): number of filter banks from stft. (Default: ``201``)
        fixed_rate (float): rate to speed up or slow down by.
            If None is provided, rate must be passed to the forward method. (Default: ``None``)
    """
    __constants__ = ['fixed_rate']

    def __init__(self, hop_length=None, n_freq=201, fixed_rate=None):
        super(TimeStretch, self).__init__()

        self.fixed_rate = fixed_rate

        n_fft = (n_freq - 1) * 2
        hop_length = hop_length if hop_length is not None else n_fft // 2
        phase_advance = torch.linspace(0, math.pi * hop_length, n_freq)[..., None]
        self.phase_advance = torch.jit.Attribute(phase_advance, torch.Tensor)

    def forward(self, complex_specgrams, overriding_rate=None):
        # type: (Tensor, Optional[float]) -> Tensor
        r"""
        Args:
            complex_specgrams (Tensor): complex spectrogram (..., freq, time, complex=2)
            overriding_rate (float or None): speed up to apply to this batch.
                If no rate is passed, use ``self.fixed_rate``

        Returns:
            (Tensor): Stretched complex spectrogram of dimension (..., freq, ceil(time/rate), complex=2)
        """
        assert complex_specgrams.size(-1) == 2, "complex_specgrams should be a complex tensor, shape (..., complex=2)"

        if overriding_rate is None:
            rate = self.fixed_rate
            if rate is None:
                raise ValueError("If no fixed_rate is specified"
                                 ", must pass a valid rate to the forward method.")
        else:
            rate = overriding_rate

        if rate == 1.0:
            return complex_specgrams

        return F.phase_vocoder(complex_specgrams, rate, self.phase_advance)


class _AxisMasking(torch.nn.Module):
    r"""Apply masking to a spectrogram.

    Args:
        mask_param (int): Maximum possible length of the mask
        axis: What dimension the mask is applied on
        iid_masks (bool): Applies iid masks to each of the examples in the batch dimension
    """
    __constants__ = ['mask_param', 'axis', 'iid_masks']

    def __init__(self, mask_param, axis, iid_masks):

        super(_AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks

    def forward(self, specgram, mask_value=0.):
        # type: (Tensor, float) -> Tensor
        r"""
        Args:
            specgram (torch.Tensor): Tensor of dimension (..., freq, time)

        Returns:
            torch.Tensor: Masked spectrogram of dimensions (..., freq, time)
        """

        # if iid_masks flag marked and specgram has a batch dimension
        if self.iid_masks and specgram.dim() == 4:
            return F.mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + 1)
        else:
            return F.mask_along_axis(specgram, self.mask_param, mask_value, self.axis)


class FrequencyMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the frequency domain.

    Args:
        freq_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, freq_mask_param).
        iid_masks (bool): weather to apply the same mask to all
            the examples/channels in the batch. (Default: False)
    """

    def __init__(self, freq_mask_param, iid_masks=False):
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)


class TimeMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the time domain.

    Args:
        time_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, time_mask_param).
        iid_masks (bool): weather to apply the same mask to all
            the examples/channels in the batch. Defaults to False.
    """

    def __init__(self, time_mask_param, iid_masks=False):
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)
