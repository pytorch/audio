# -*- coding: utf-8 -*-

import math
import warnings
from typing import Callable, Optional

import torch
from torch import Tensor
from torchaudio import functional as F

from .functional.functional import (
    _get_sinc_resample_kernel,
    _apply_sinc_resample_kernel,
)

__all__ = [
    'Spectrogram',
    'InverseSpectrogram',
    'GriffinLim',
    'AmplitudeToDB',
    'MelScale',
    'InverseMelScale',
    'MelSpectrogram',
    'MFCC',
    'LFCC',
    'MuLawEncoding',
    'MuLawDecoding',
    'Resample',
    'ComplexNorm',
    'TimeStretch',
    'Fade',
    'FrequencyMasking',
    'TimeMasking',
    'SlidingWindowCmn',
    'Vad',
    'SpectralCentroid',
    'Vol',
    'ComputeDeltas',
    'PitchShift',
]


class Spectrogram(torch.nn.Module):
    r"""Create a spectrogram from a audio signal.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float or None, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy (Default: ``True``)
        return_complex (bool, optional):
            Indicates whether the resulting complex-valued Tensor should be represented with
            native complex dtype, such as `torch.cfloat` and `torch.cdouble`, or real dtype
            mimicking complex value with an extra dimension for real and imaginary parts.
            (See also ``torch.view_as_real``.)
            This argument is only effective when ``power=None``. It is ignored for
            cases where ``power`` is a number as in those cases, the returned tensor is
            power spectrogram, which is a real-valued tensor.

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = torchaudio.transforms.Spectrogram(n_fft=800)
        >>> spectrogram = transform(waveform)

    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 return_complex: bool = True) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_complex = return_complex

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Dimension (..., freq, time), where freq is
            ``n_fft // 2 + 1`` where ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
        return F.spectrogram(
            waveform,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
            self.return_complex,
        )


class InverseSpectrogram(torch.nn.Module):
    r"""Create an inverse spectrogram to recover an audio signal from a spectrogram.

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (None, optional): Required to be None to match the case where Spectrogram transform returns
            a full complex spectrogram.
        normalized (bool, optional): Whether the spectrogram was normalized by magnitude after stft.
            (Default: ``False``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether the signal in spectrogram was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether spectrogram was used to return half of results to
            avoid redundancy (Default: ``True``)
        return_complex (bool, optional):
            This value is ignored, and a real-valued output is always returned. It is provided only
            for compatibility with the Spectrogram transform.
            Default: ``False``
    """
    __constants__ = ['n_fft', 'win_length', 'hop_length', 'pad', 'power', 'normalized']

    def __init__(self,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: Optional[float] = None,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 return_complex: bool = False) -> None:
        super(InverseSpectrogram, self).__init__()
        self.n_fft = n_fft
        # number of FFT bins. the returned STFT result will have n_fft // 2 + 1
        # number of frequencies due to onesided=True in torch.stft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided
        self.return_complex = return_complex

    def forward(self, spectrogram: Tensor, length: Optional[int] = None) -> Tensor:
        r"""
        Args:
            spectrogram (Tensor): Complex tensor of audio of dimension (..., freq, time).
                Alternatively, the tensor can be pseudo-complex, i.e. a real tensor of
                dimension (..., freq, time, 2), if return_complex is False, although this
                behavior is deprecated.
            length (int, optional): The amount to trim the signal by (i.e. the original signal length).
                Default: ``None`` (whole signal).

        Returns:
            Tensor: Dimension (..., time), Least squares estimation of the original signal.
        """
        return F.inverse_spectrogram(
            spectrogram,
            length,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.power,
            self.normalized,
            self.center,
            self.pad_mode,
            self.onesided,
            self.return_complex,
        )


class GriffinLim(torch.nn.Module):
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

    Implementation ported from
    *librosa* [:footcite:`brian_mcfee-proc-scipy-2015`], *A fast Griffin-Lim algorithm* [:footcite:`6701851`]
    and *Signal estimation from modified short-time Fourier transform* [:footcite:`1172092`].

    Args:
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        n_iter (int, optional): Number of iteration for phase recovery process. (Default: ``32``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)
        momentum (float, optional): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge. (Default: ``0.99``)
        length (int, optional): Array length of the expected output. (Default: ``None``)
        rand_init (bool, optional): Initializes phase randomly if True and to zero otherwise. (Default: ``True``)

    Example
        >>> batch, freq, time = 2, 257, 100
        >>> spectrogram = torch.randn(batch, freq, time)
        >>> transform = transforms.GriffinLim(n_fft=512)
        >>> waveform = transform(spectrogram)
    """
    __constants__ = ['n_fft', 'n_iter', 'win_length', 'hop_length', 'power',
                     'length', 'momentum', 'rand_init']

    def __init__(self,
                 n_fft: int = 400,
                 n_iter: int = 32,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: float = 2.,
                 wkwargs: Optional[dict] = None,
                 momentum: float = 0.99,
                 length: Optional[int] = None,
                 rand_init: bool = True) -> None:
        super(GriffinLim, self).__init__()

        assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
        assert momentum >= 0, 'momentum={} < 0'.format(momentum)

        self.n_fft = n_fft
        self.n_iter = n_iter
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.length = length
        self.power = power
        self.momentum = momentum / (1 + momentum)
        self.rand_init = rand_init

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor):
                A magnitude-only STFT spectrogram of dimension (..., freq, frames)
                where freq is ``n_fft // 2 + 1``.

        Returns:
            Tensor: waveform of (..., time), where time equals the ``length`` parameter if given.
        """
        return F.griffinlim(specgram, self.window, self.n_fft, self.hop_length, self.win_length, self.power,
                            self.n_iter, self.momentum, self.length, self.rand_init)


class AmplitudeToDB(torch.nn.Module):
    r"""Turn a tensor from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        stype (str, optional): scale of input tensor ('power' or 'magnitude'). The
            power being the elementwise square of the magnitude. (Default: ``'power'``)
        top_db (float, optional): minimum negative cut-off in decibels.  A reasonable number
            is 80. (Default: ``None``)
    """
    __constants__ = ['multiplier', 'amin', 'ref_value', 'db_multiplier']

    def __init__(self, stype: str = 'power', top_db: Optional[float] = None) -> None:
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError('top_db must be positive value')
        self.top_db = top_db
        self.multiplier = 10.0 if stype == 'power' else 20.0
        self.amin = 1e-10
        self.ref_value = 1.0
        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

    def forward(self, x: Tensor) -> Tensor:
        r"""Numerically stable implementation from Librosa.

        https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html

        Args:
            x (Tensor): Input tensor before being converted to decibel scale.

        Returns:
            Tensor: Output tensor in decibel scale.
        """
        return F.amplitude_to_DB(x, self.multiplier, self.amin, self.db_multiplier, self.top_db)


class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`. (Default: ``201``)
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: int = 201,
                 norm: Optional[str] = None,
                 mel_scale: str = "htk") -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.norm = norm
        self.mel_scale = mel_scale

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)
        fb = F.melscale_fbanks(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.norm,
            self.mel_scale)
        self.register_buffer('fb', fb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # (..., time, freq) dot (freq, n_mels) -> (..., n_mels, time)
        mel_specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

        return mel_specgram


class InverseMelScale(torch.nn.Module):
    r"""Solve for a normal STFT from a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    It minimizes the euclidian norm between the input mel-spectrogram and the product between
    the estimated spectrogram and the filter banks using SGD.

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        max_iter (int, optional): Maximum number of optimization iterations. (Default: ``100000``)
        tolerance_loss (float, optional): Value of loss to stop optimization at. (Default: ``1e-5``)
        tolerance_change (float, optional): Difference in losses to stop optimization at. (Default: ``1e-8``)
        sgdargs (dict or None, optional): Arguments for the SGD optimizer. (Default: ``None``)
        norm (Optional[str]): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    """
    __constants__ = ['n_stft', 'n_mels', 'sample_rate', 'f_min', 'f_max', 'max_iter', 'tolerance_loss',
                     'tolerance_change', 'sgdargs']

    def __init__(self,
                 n_stft: int,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 max_iter: int = 100000,
                 tolerance_loss: float = 1e-5,
                 tolerance_change: float = 1e-8,
                 sgdargs: Optional[dict] = None,
                 norm: Optional[str] = None,
                 mel_scale: str = "htk") -> None:
        super(InverseMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min
        self.max_iter = max_iter
        self.tolerance_loss = tolerance_loss
        self.tolerance_change = tolerance_change
        self.sgdargs = sgdargs or {'lr': 0.1, 'momentum': 0.9}

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)

        fb = F.melscale_fbanks(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate,
                               norm, mel_scale)
        self.register_buffer('fb', fb)

    def forward(self, melspec: Tensor) -> Tensor:
        r"""
        Args:
            melspec (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)

        Returns:
            Tensor: Linear scale spectrogram of size (..., freq, time)
        """
        # pack batch
        shape = melspec.size()
        melspec = melspec.view(-1, shape[-2], shape[-1])

        n_mels, time = shape[-2], shape[-1]
        freq, _ = self.fb.size()  # (freq, n_mels)
        melspec = melspec.transpose(-1, -2)
        assert self.n_mels == n_mels

        specgram = torch.rand(melspec.size()[0], time, freq, requires_grad=True,
                              dtype=melspec.dtype, device=melspec.device)

        optim = torch.optim.SGD([specgram], **self.sgdargs)

        loss = float('inf')
        for _ in range(self.max_iter):
            optim.zero_grad()
            diff = melspec - specgram.matmul(self.fb)
            new_loss = diff.pow(2).sum(axis=-1).mean()
            # take sum over mel-frequency then average over other dimensions
            # so that loss threshold is applied par unit timeframe
            new_loss.backward()
            optim.step()
            specgram.data = specgram.data.clamp(min=0)

            new_loss = new_loss.item()
            if new_loss < self.tolerance_loss or abs(loss - new_loss) < self.tolerance_change:
                break
            loss = new_loss

        specgram.requires_grad_(False)
        specgram = specgram.clamp(min=0).transpose(-1, -2)

        # unpack batch
        specgram = specgram.view(shape[:-2] + (freq, time))
        return specgram


class MelSpectrogram(torch.nn.Module):
    r"""Create MelSpectrogram for a raw audio signal. This is a composition of Spectrogram
    and MelScale.

    Sources
        * https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe
        * https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
        * http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        power (float, optional): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc. (Default: ``2``)
        normalized (bool, optional): Whether to normalize by magnitude after stft. (Default: ``False``)
        wkwargs (Dict[..., ...] or None, optional): Arguments for window function. (Default: ``None``)
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. (Default: ``"reflect"``)
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. (Default: ``True``)
        norm (Optional[str]): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = transforms.MelSpectrogram(sample_rate)
        >>> mel_specgram = transform(waveform)  # (channel, n_mels, time)
    """
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad', 'n_mels', 'f_min']

    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 pad: int = 0,
                 n_mels: int = 128,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 power: float = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None,
                 center: bool = True,
                 pad_mode: str = "reflect",
                 onesided: bool = True,
                 norm: Optional[str] = None,
                 mel_scale: str = "htk") -> None:
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       pad=self.pad, window_fn=window_fn, power=self.power,
                                       normalized=self.normalized, wkwargs=wkwargs,
                                       center=center, pad_mode=pad_mode, onesided=onesided)
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            norm,
            mel_scale
        )

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveform)
        mel_specgram = self.mel_scale(specgram)
        return mel_specgram


class MFCC(torch.nn.Module):
    r"""Create the Mel-frequency cepstrum coefficients from an audio signal.

    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_mfcc (int, optional): Number of mfc coefficients to retain. (Default: ``40``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``'ortho'``)
        log_mels (bool, optional): whether to use log-mel spectrograms instead of db-scaled. (Default: ``False``)
        melkwargs (dict or None, optional): arguments for MelSpectrogram. (Default: ``None``)
    """
    __constants__ = ['sample_rate', 'n_mfcc', 'dct_type', 'top_db', 'log_mels']

    def __init__(self,
                 sample_rate: int = 16000,
                 n_mfcc: int = 40,
                 dct_type: int = 2,
                 norm: str = 'ortho',
                 log_mels: bool = False,
                 melkwargs: Optional[dict] = None) -> None:
        super(MFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported: {}'.format(dct_type))
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)

        melkwargs = melkwargs or {}
        self.MelSpectrogram = MelSpectrogram(sample_rate=self.sample_rate, **melkwargs)

        if self.n_mfcc > self.MelSpectrogram.n_mels:
            raise ValueError('Cannot select more MFCC coefficients than # mel bins')
        dct_mat = F.create_dct(self.n_mfcc, self.MelSpectrogram.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_mels = log_mels

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: specgram_mel_db of size (..., ``n_mfcc``, time).
        """
        mel_specgram = self.MelSpectrogram(waveform)
        if self.log_mels:
            log_offset = 1e-6
            mel_specgram = torch.log(mel_specgram + log_offset)
        else:
            mel_specgram = self.amplitude_to_DB(mel_specgram)

        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_nfcc, time)
        mfcc = torch.matmul(mel_specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc


class LFCC(torch.nn.Module):
    r"""Create the linear-frequency cepstrum coefficients from an audio signal.

    By default, this calculates the LFCC on the DB-scaled linear filtered spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        n_filter (int, optional): Number of linear filters to apply. (Default: ``128``)
        n_lfcc (int, optional): Number of lfc coefficients to retain. (Default: ``40``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``None``)
        dct_type (int, optional): type of DCT (discrete cosine transform) to use. (Default: ``2``)
        norm (str, optional): norm to use. (Default: ``'ortho'``)
        log_lf (bool, optional): whether to use log-lf spectrograms instead of db-scaled. (Default: ``False``)
        speckwargs (dict or None, optional): arguments for Spectrogram. (Default: ``None``)
    """
    __constants__ = ['sample_rate', 'n_filter', 'n_lfcc', 'dct_type', 'top_db', 'log_lf']

    def __init__(self,
                 sample_rate: int = 16000,
                 n_filter: int = 128,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_lfcc: int = 40,
                 dct_type: int = 2,
                 norm: str = 'ortho',
                 log_lf: bool = False,
                 speckwargs: Optional[dict] = None) -> None:
        super(LFCC, self).__init__()
        supported_dct_types = [2]
        if dct_type not in supported_dct_types:
            raise ValueError('DCT type not supported: {}'.format(dct_type))
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.dct_type = dct_type
        self.norm = norm
        self.top_db = 80.0
        self.amplitude_to_DB = AmplitudeToDB('power', self.top_db)

        speckwargs = speckwargs or {}
        self.Spectrogram = Spectrogram(**speckwargs)

        if self.n_lfcc > self.Spectrogram.n_fft:
            raise ValueError('Cannot select more LFCC coefficients than # fft bins')

        filter_mat = F.linear_fbanks(
            n_freqs=self.Spectrogram.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_filter=self.n_filter,
            sample_rate=self.sample_rate,
        )
        self.register_buffer("filter_mat", filter_mat)

        dct_mat = F.create_dct(self.n_lfcc, self.n_filter, self.norm)
        self.register_buffer('dct_mat', dct_mat)
        self.log_lf = log_lf

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Linear Frequency Cepstral Coefficients of size (..., ``n_lfcc``, time).
        """
        specgram = self.Spectrogram(waveform)

        # (..., time, freq) dot (freq, n_filter) -> (..., n_filter, time)
        specgram = torch.matmul(specgram.transpose(-1, -2), self.filter_mat).transpose(-1, -2)

        if self.log_lf:
            log_offset = 1e-6
            specgram = torch.log(specgram + log_offset)
        else:
            specgram = self.amplitude_to_DB(specgram)

        # (..., time, n_filter) dot (n_filter, n_lfcc) -> (..., n_lfcc, time)
        lfcc = torch.matmul(specgram.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return lfcc


class MuLawEncoding(torch.nn.Module):
    r"""Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Args:
        quantization_channels (int, optional): Number of channels. (Default: ``256``)

    Example
       >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
       >>> transform = torchaudio.transforms.MuLawEncoding(quantization_channels=512)
       >>> mulawtrans = transform(waveform)

    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels: int = 256) -> None:
        super(MuLawEncoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): A signal to be encoded.

        Returns:
            x_mu (Tensor): An encoded signal.
        """
        return F.mu_law_encoding(x, self.quantization_channels)


class MuLawDecoding(torch.nn.Module):
    r"""Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        quantization_channels (int, optional): Number of channels. (Default: ``256``)

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = torchaudio.transforms.MuLawDecoding(quantization_channels=512)
        >>> mulawtrans = transform(waveform)
    """
    __constants__ = ['quantization_channels']

    def __init__(self, quantization_channels: int = 256) -> None:
        super(MuLawDecoding, self).__init__()
        self.quantization_channels = quantization_channels

    def forward(self, x_mu: Tensor) -> Tensor:
        r"""
        Args:
            x_mu (Tensor): A mu-law encoded signal which needs to be decoded.

        Returns:
            Tensor: The signal decoded.
        """
        return F.mu_law_decoding(x_mu, self.quantization_channels)


class Resample(torch.nn.Module):
    r"""Resample a signal from one frequency to another. A resampling method can be given.

    Note:
        If resampling on waveforms of higher precision than float32, there may be a small loss of precision
        because the kernel is cached once as float32. If high precision resampling is important for your application,
        the functional form will retain higher precision, but run slower because it does not cache the kernel.
        Alternatively, you could rewrite a transform that caches a higher precision kernel.

    Args:
        orig_freq (float, optional): The original frequency of the signal. (Default: ``16000``)
        new_freq (float, optional): The desired frequency. (Default: ``16000``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interpolation``, ``kaiser_window``] (Default: ``'sinc_interpolation'``)
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        beta (float or None): The shape parameter used for kaiser window.
        dtype (torch.device, optional):
            Determnines the precision that resampling kernel is pre-computed and cached. If not provided,
            kernel is computed with ``torch.float64`` then cached as ``torch.float32``.
            If you need higher precision, provide ``torch.float64``, and the pre-computed kernel is computed and
            cached as ``torch.float64``. If you use resample with lower precision, then instead of providing this
            providing this argument, please use ``Resample.to(dtype)``, so that the kernel generation is still
            carried out on ``torch.float64``.

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = transforms.Resample(sample_rate, sample_rate/10)
        >>> waveform = transform(waveform)
    """

    def __init__(
            self,
            orig_freq: float = 16000,
            new_freq: float = 16000,
            resampling_method: str = 'sinc_interpolation',
            lowpass_filter_width: int = 6,
            rolloff: float = 0.99,
            beta: Optional[float] = None,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = math.gcd(int(self.orig_freq), int(self.new_freq))
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta

        if self.orig_freq != self.new_freq:
            kernel, self.width = _get_sinc_resample_kernel(
                self.orig_freq, self.new_freq, self.gcd,
                self.lowpass_filter_width, self.rolloff,
                self.resampling_method, beta, dtype=dtype)
            self.register_buffer('kernel', kernel)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Output signal of dimension (..., time).
        """
        if self.orig_freq == self.new_freq:
            return waveform
        return _apply_sinc_resample_kernel(
            waveform, self.orig_freq, self.new_freq, self.gcd,
            self.kernel, self.width)


class ComplexNorm(torch.nn.Module):
    r"""Compute the norm of complex tensor input.

    Args:
        power (float, optional): Power of the norm. (Default: to ``1.0``)

    Example
        >>> complex_tensor = ... #  Tensor shape of (…, complex=2)
        >>> transform = transforms.ComplexNorm(power=2)
        >>> complex_norm = transform(complex_tensor)
    """
    __constants__ = ['power']

    def __init__(self, power: float = 1.0) -> None:
        warnings.warn(
            'torchaudio.transforms.ComplexNorm has been deprecated '
            'and will be removed from future release.'
            'Please convert the input Tensor to complex type with `torch.view_as_complex` then '
            'use `torch.abs` and `torch.angle`. '
            'Please refer to https://github.com/pytorch/audio/issues/1337 '
            "for more details about torchaudio's plan to migrate to native complex type."
        )
        super(ComplexNorm, self).__init__()
        self.power = power

    def forward(self, complex_tensor: Tensor) -> Tensor:
        r"""
        Args:
            complex_tensor (Tensor): Tensor shape of `(..., complex=2)`.

        Returns:
            Tensor: norm of the input tensor, shape of `(..., )`.
        """
        return F.complex_norm(complex_tensor, self.power)


class ComputeDeltas(torch.nn.Module):
    r"""Compute delta coefficients of a tensor, usually a spectrogram.

    See `torchaudio.functional.compute_deltas` for more details.

    Args:
        win_length (int): The window length used for computing delta. (Default: ``5``)
        mode (str): Mode parameter passed to padding. (Default: ``'replicate'``)
    """
    __constants__ = ['win_length']

    def __init__(self, win_length: int = 5, mode: str = "replicate") -> None:
        super(ComputeDeltas, self).__init__()
        self.win_length = win_length
        self.mode = mode

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): Tensor of audio of dimension (..., freq, time).

        Returns:
            Tensor: Tensor of deltas of dimension (..., freq, time).
        """
        return F.compute_deltas(specgram, win_length=self.win_length, mode=self.mode)


class TimeStretch(torch.nn.Module):
    r"""Stretch stft in time without modifying pitch for a given rate.

    Args:
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        n_freq (int, optional): number of filter banks from stft. (Default: ``201``)
        fixed_rate (float or None, optional): rate to speed up or slow down by.
            If None is provided, rate must be passed to the forward method. (Default: ``None``)
    """
    __constants__ = ['fixed_rate']

    def __init__(self,
                 hop_length: Optional[int] = None,
                 n_freq: int = 201,
                 fixed_rate: Optional[float] = None) -> None:
        super(TimeStretch, self).__init__()

        self.fixed_rate = fixed_rate

        n_fft = (n_freq - 1) * 2
        hop_length = hop_length if hop_length is not None else n_fft // 2
        self.register_buffer('phase_advance', torch.linspace(0, math.pi * hop_length, n_freq)[..., None])

    def forward(self, complex_specgrams: Tensor, overriding_rate: Optional[float] = None) -> Tensor:
        r"""
        Args:
            complex_specgrams (Tensor):
                Either a real tensor of dimension of ``(..., freq, num_frame, complex=2)``
                or a tensor of dimension ``(..., freq, num_frame)`` with complex dtype.
            overriding_rate (float or None, optional): speed up to apply to this batch.
                If no rate is passed, use ``self.fixed_rate``. (Default: ``None``)

        Returns:
            Tensor:
                Stretched spectrogram. The resulting tensor is of the same dtype as the input
                spectrogram, but the number of frames is changed to ``ceil(num_frame / rate)``.
        """
        if overriding_rate is None:
            if self.fixed_rate is None:
                raise ValueError(
                    "If no fixed_rate is specified, must pass a valid rate to the forward method.")
            rate = self.fixed_rate
        else:
            rate = overriding_rate
        return F.phase_vocoder(complex_specgrams, rate, self.phase_advance)


class Fade(torch.nn.Module):
    r"""Add a fade in and/or fade out to an waveform.

    Args:
        fade_in_len (int, optional): Length of fade-in (time frames). (Default: ``0``)
        fade_out_len (int, optional): Length of fade-out (time frames). (Default: ``0``)
        fade_shape (str, optional): Shape of fade. Must be one of: "quarter_sine",
            "half_sine", "linear", "logarithmic", "exponential". (Default: ``"linear"``)
    """

    def __init__(self,
                 fade_in_len: int = 0,
                 fade_out_len: int = 0,
                 fade_shape: str = "linear") -> None:
        super(Fade, self).__init__()
        self.fade_in_len = fade_in_len
        self.fade_out_len = fade_out_len
        self.fade_shape = fade_shape

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Tensor of audio of dimension (..., time).
        """
        waveform_length = waveform.size()[-1]
        device = waveform.device
        return self._fade_in(waveform_length).to(device) * \
            self._fade_out(waveform_length).to(device) * waveform

    def _fade_in(self, waveform_length: int) -> Tensor:
        fade = torch.linspace(0, 1, self.fade_in_len)
        ones = torch.ones(waveform_length - self.fade_in_len)

        if self.fade_shape == "linear":
            fade = fade

        if self.fade_shape == "exponential":
            fade = torch.pow(2, (fade - 1)) * fade

        if self.fade_shape == "logarithmic":
            fade = torch.log10(.1 + fade) + 1

        if self.fade_shape == "quarter_sine":
            fade = torch.sin(fade * math.pi / 2)

        if self.fade_shape == "half_sine":
            fade = torch.sin(fade * math.pi - math.pi / 2) / 2 + 0.5

        return torch.cat((fade, ones)).clamp_(0, 1)

    def _fade_out(self, waveform_length: int) -> Tensor:
        fade = torch.linspace(0, 1, self.fade_out_len)
        ones = torch.ones(waveform_length - self.fade_out_len)

        if self.fade_shape == "linear":
            fade = - fade + 1

        if self.fade_shape == "exponential":
            fade = torch.pow(2, - fade) * (1 - fade)

        if self.fade_shape == "logarithmic":
            fade = torch.log10(1.1 - fade) + 1

        if self.fade_shape == "quarter_sine":
            fade = torch.sin(fade * math.pi / 2 + math.pi / 2)

        if self.fade_shape == "half_sine":
            fade = torch.sin(fade * math.pi + math.pi / 2) / 2 + 0.5

        return torch.cat((ones, fade)).clamp_(0, 1)


class _AxisMasking(torch.nn.Module):
    r"""Apply masking to a spectrogram.

    Args:
        mask_param (int): Maximum possible length of the mask.
        axis (int): What dimension the mask is applied on.
        iid_masks (bool): Applies iid masks to each of the examples in the batch dimension.
            This option is applicable only when the input tensor is 4D.
    """
    __constants__ = ['mask_param', 'axis', 'iid_masks']

    def __init__(self, mask_param: int, axis: int, iid_masks: bool) -> None:

        super(_AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks

    def forward(self, specgram: Tensor, mask_value: float = 0.) -> Tensor:
        r"""
        Args:
            specgram (Tensor): Tensor of dimension (..., freq, time).
            mask_value (float): Value to assign to the masked columns.

        Returns:
            Tensor: Masked spectrogram of dimensions (..., freq, time).
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
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.
    """

    def __init__(self, freq_mask_param: int, iid_masks: bool = False) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)


class TimeMasking(_AxisMasking):
    r"""Apply masking to a spectrogram in the time domain.

    Args:
        time_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, time_mask_param).
        iid_masks (bool, optional): whether to apply different masks to each
            example/channel in the batch. (Default: ``False``)
            This option is applicable only when the input tensor is 4D.
    """

    def __init__(self, time_mask_param: int, iid_masks: bool = False) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)


class Vol(torch.nn.Module):
    r"""Add a volume to an waveform.

    Args:
        gain (float): Interpreted according to the given gain_type:
            If ``gain_type`` = ``amplitude``, ``gain`` is a positive amplitude ratio.
            If ``gain_type`` = ``power``, ``gain`` is a power (voltage squared).
            If ``gain_type`` = ``db``, ``gain`` is in decibels.
        gain_type (str, optional): Type of gain. One of: ``amplitude``, ``power``, ``db`` (Default: ``amplitude``)
    """

    def __init__(self, gain: float, gain_type: str = 'amplitude'):
        super(Vol, self).__init__()
        self.gain = gain
        self.gain_type = gain_type

        if gain_type in ['amplitude', 'power'] and gain < 0:
            raise ValueError("If gain_type = amplitude or power, gain must be positive.")

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Tensor of audio of dimension (..., time).
        """
        if self.gain_type == "amplitude":
            waveform = waveform * self.gain

        if self.gain_type == "db":
            waveform = F.gain(waveform, self.gain)

        if self.gain_type == "power":
            waveform = F.gain(waveform, 10 * math.log10(self.gain))

        return torch.clamp(waveform, -1, 1)


class SlidingWindowCmn(torch.nn.Module):
    r"""
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    Args:
        cmn_window (int, optional): Window in frames for running average CMN computation (int, default = 600)
        min_cmn_window (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center == false, ignored if center==true (int, default = 100)
        center (bool, optional): If true, use a window centered on the current frame
            (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
        norm_vars (bool, optional): If true, normalize variance to one. (bool, default = false)
    """

    def __init__(self,
                 cmn_window: int = 600,
                 min_cmn_window: int = 100,
                 center: bool = False,
                 norm_vars: bool = False) -> None:
        super().__init__()
        self.cmn_window = cmn_window
        self.min_cmn_window = min_cmn_window
        self.center = center
        self.norm_vars = norm_vars

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Tensor of audio of dimension (..., time).
        """
        cmn_waveform = F.sliding_window_cmn(
            waveform, self.cmn_window, self.min_cmn_window, self.center, self.norm_vars)
        return cmn_waveform


class Vad(torch.nn.Module):
    r"""Voice Activity Detector. Similar to SoX implementation.
    Attempts to trim silence and quiet background sounds from the ends of recordings of speech.
    The algorithm currently uses a simple cepstral power measurement to detect voice,
    so may be fooled by other things, especially music.

    The effect can trim only from the front of the audio,
    so in order to trim from the back, the reverse effect must also be used.

    Args:
        sample_rate (int): Sample rate of audio signal.
        trigger_level (float, optional): The measurement level used to trigger activity detection.
            This may need to be cahnged depending on the noise level, signal level,
            and other characteristics of the input audio. (Default: 7.0)
        trigger_time (float, optional): The time constant (in seconds)
            used to help ignore short bursts of sound. (Default: 0.25)
        search_time (float, optional): The amount of audio (in seconds)
            to search for quieter/shorter bursts of audio to include prior
            to the detected trigger point. (Default: 1.0)
        allowed_gap (float, optional): The allowed gap (in seconds) between
            quiteter/shorter bursts of audio to include prior
            to the detected trigger point. (Default: 0.25)
        pre_trigger_time (float, optional): The amount of audio (in seconds) to preserve
            before the trigger point and any found quieter/shorter bursts. (Default: 0.0)
        boot_time (float, optional) The algorithm (internally) uses adaptive noise
            estimation/reduction in order to detect the start of the wanted audio.
            This option sets the time for the initial noise estimate. (Default: 0.35)
        noise_up_time (float, optional) Time constant used by the adaptive noise estimator
            for when the noise level is increasing. (Default: 0.1)
        noise_down_time (float, optional) Time constant used by the adaptive noise estimator
            for when the noise level is decreasing. (Default: 0.01)
        noise_reduction_amount (float, optional) Amount of noise reduction to use in
            the detection algorithm (e.g. 0, 0.5, ...). (Default: 1.35)
        measure_freq (float, optional) Frequency of the algorithm’s
            processing/measurements. (Default: 20.0)
        measure_duration: (float, optional) Measurement duration.
            (Default: Twice the measurement period; i.e. with overlap.)
        measure_smooth_time (float, optional) Time constant used to smooth
            spectral measurements. (Default: 0.4)
        hp_filter_freq (float, optional) "Brick-wall" frequency of high-pass filter applied
            at the input to the detector algorithm. (Default: 50.0)
        lp_filter_freq (float, optional) "Brick-wall" frequency of low-pass filter applied
            at the input to the detector algorithm. (Default: 6000.0)
        hp_lifter_freq (float, optional) "Brick-wall" frequency of high-pass lifter used
            in the detector algorithm. (Default: 150.0)
        lp_lifter_freq (float, optional) "Brick-wall" frequency of low-pass lifter used
            in the detector algorithm. (Default: 2000.0)

    Reference:
        - http://sox.sourceforge.net/sox.html
    """

    def __init__(self,
                 sample_rate: int,
                 trigger_level: float = 7.0,
                 trigger_time: float = 0.25,
                 search_time: float = 1.0,
                 allowed_gap: float = 0.25,
                 pre_trigger_time: float = 0.0,
                 boot_time: float = .35,
                 noise_up_time: float = .1,
                 noise_down_time: float = .01,
                 noise_reduction_amount: float = 1.35,
                 measure_freq: float = 20.0,
                 measure_duration: Optional[float] = None,
                 measure_smooth_time: float = .4,
                 hp_filter_freq: float = 50.,
                 lp_filter_freq: float = 6000.,
                 hp_lifter_freq: float = 150.,
                 lp_lifter_freq: float = 2000.) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.trigger_level = trigger_level
        self.trigger_time = trigger_time
        self.search_time = search_time
        self.allowed_gap = allowed_gap
        self.pre_trigger_time = pre_trigger_time
        self.boot_time = boot_time
        self.noise_up_time = noise_up_time
        self.noise_down_time = noise_down_time
        self.noise_reduction_amount = noise_reduction_amount
        self.measure_freq = measure_freq
        self.measure_duration = measure_duration
        self.measure_smooth_time = measure_smooth_time
        self.hp_filter_freq = hp_filter_freq
        self.lp_filter_freq = lp_filter_freq
        self.hp_lifter_freq = hp_lifter_freq
        self.lp_lifter_freq = lp_lifter_freq

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension `(channels, time)` or `(time)`
                Tensor of shape `(channels, time)` is treated as a multi-channel recording
                of the same event and the resulting output will be trimmed to the earliest
                voice activity in any channel.
        """
        return F.vad(
            waveform=waveform,
            sample_rate=self.sample_rate,
            trigger_level=self.trigger_level,
            trigger_time=self.trigger_time,
            search_time=self.search_time,
            allowed_gap=self.allowed_gap,
            pre_trigger_time=self.pre_trigger_time,
            boot_time=self.boot_time,
            noise_up_time=self.noise_up_time,
            noise_down_time=self.noise_down_time,
            noise_reduction_amount=self.noise_reduction_amount,
            measure_freq=self.measure_freq,
            measure_duration=self.measure_duration,
            measure_smooth_time=self.measure_smooth_time,
            hp_filter_freq=self.hp_filter_freq,
            lp_filter_freq=self.lp_filter_freq,
            hp_lifter_freq=self.hp_lifter_freq,
            lp_lifter_freq=self.lp_lifter_freq,
        )


class SpectralCentroid(torch.nn.Module):
    r"""Compute the spectral centroid for each channel along the time axis.

    The spectral centroid is defined as the weighted average of the
    frequency values, weighted by their magnitude.

    Args:
        sample_rate (int): Sample rate of audio signal.
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int or None, optional): Window size. (Default: ``n_fft``)
        hop_length (int or None, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
        pad (int, optional): Two sided padding of signal. (Default: ``0``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        wkwargs (dict or None, optional): Arguments for window function. (Default: ``None``)

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = transforms.SpectralCentroid(sample_rate)
        >>> spectral_centroid = transform(waveform)  # (channel, time)
    """
    __constants__ = ['sample_rate', 'n_fft', 'win_length', 'hop_length', 'pad']

    def __init__(self,
                 sample_rate: int,
                 n_fft: int = 400,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 wkwargs: Optional[dict] = None) -> None:
        super(SpectralCentroid, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Spectral Centroid of size (..., time).
        """

        return F.spectral_centroid(waveform, self.sample_rate, self.pad, self.window, self.n_fft, self.hop_length,
                                   self.win_length)


class PitchShift(torch.nn.Module):
    r"""Shift the pitch of a waveform by ``n_steps`` steps.

    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (float): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default : ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then ``win_length // 4``
            is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).

    Example
        >>> waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
        >>> transform = transforms.PitchShift(sample_rate, 4)
        >>> waveform_shift = transform(waveform)  # (channel, time)
    """
    __constants__ = ['sample_rate', 'n_steps', 'bins_per_octave', 'n_fft', 'win_length', 'hop_length']

    def __init__(self,
                 sample_rate: int,
                 n_steps: int,
                 bins_per_octave: int = 12,
                 n_fft: int = 512,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 window_fn: Callable[..., Tensor] = torch.hann_window,
                 wkwargs: Optional[dict] = None) -> None:
        super(PitchShift, self).__init__()
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 4
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: The pitch-shifted audio of shape `(..., time)`.
        """

        return F.pitch_shift(waveform, self.sample_rate, self.n_steps, self.bins_per_octave, self.n_fft,
                             self.win_length, self.hop_length, self.window)
