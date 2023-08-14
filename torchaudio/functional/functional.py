# -*- coding: utf-8 -*-

import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated

from .filtering import highpass_biquad, treble_biquad

__all__ = [
    "spectrogram",
    "inverse_spectrogram",
    "griffinlim",
    "amplitude_to_DB",
    "DB_to_amplitude",
    "compute_deltas",
    "melscale_fbanks",
    "linear_fbanks",
    "create_dct",
    "compute_deltas",
    "detect_pitch_frequency",
    "DB_to_amplitude",
    "mu_law_encoding",
    "mu_law_decoding",
    "phase_vocoder",
    "mask_along_axis",
    "mask_along_axis_iid",
    "sliding_window_cmn",
    "spectral_centroid",
    "apply_codec",
    "resample",
    "edit_distance",
    "loudness",
    "pitch_shift",
    "rnnt_loss",
    "psd",
    "mvdr_weights_souden",
    "mvdr_weights_rtf",
    "rtf_evd",
    "rtf_power",
    "apply_beamforming",
    "fftconvolve",
    "convolve",
    "add_noise",
    "speed",
    "preemphasis",
    "deemphasis",
]


def spectrogram(
    waveform: Tensor,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: Optional[float],
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    return_complex: Optional[bool] = None,
) -> Tensor:
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
    The spectrogram can be either magnitude-only or complex.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., time)`
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float or None): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool or str): Whether to normalize by magnitude after stft. If input is str, choices are
            ``"window"`` and ``"frame_length"``, if specific normalization type is desirable. ``True`` maps to
            ``"window"``. When normalized on ``"window"``, waveform is normalized upon the window's L2 energy. If
            normalized on ``"frame_length"``, waveform is normalized by dividing by
            :math:`(\text{frame\_length})^{0.5}`.
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. Default: ``True``
        return_complex (bool, optional):
            Deprecated and not used.

    Returns:
        Tensor: Dimension `(..., freq, time)`, freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
    """
    if return_complex is not None:
        warnings.warn(
            "`return_complex` argument is now deprecated and is not effective."
            "`torchaudio.functional.spectrogram(power=None)` always returns a tensor with "
            "complex dtype. Please remove the argument in the function call."
        )

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=frame_length_norm,
        onesided=onesided,
        return_complex=True,
    )

    # unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

    if window_norm:
        spec_f /= window.pow(2.0).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    return spec_f


def inverse_spectrogram(
    spectrogram: Tensor,
    length: Optional[int],
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    normalized: Union[bool, str],
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
) -> Tensor:
    r"""Create an inverse spectrogram or a batch of inverse spectrograms from the provided
    complex-valued spectrogram.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        spectrogram (Tensor): Complex tensor of audio of dimension (..., freq, time).
        length (int or None): The output length of the waveform.
        pad (int): Two sided padding of signal. It is only effective when ``length`` is provided.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        normalized (bool or str): Whether the stft output was normalized by magnitude. If input is str, choices are
            ``"window"`` and ``"frame_length"``, dependent on normalization mode. ``True`` maps to
            ``"window"``.
        center (bool, optional): whether the waveform was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. This parameter is provided for compatibility with the
            spectrogram function and is not used. Default: ``"reflect"``
        onesided (bool, optional): controls whether spectrogram was done in onesided mode.
            Default: ``True``

    Returns:
        Tensor: Dimension `(..., time)`. Least squares estimation of the original signal.
    """

    frame_length_norm, window_norm = _get_spec_norms(normalized)

    if not spectrogram.is_complex():
        raise ValueError("Expected `spectrogram` to be complex dtype.")

    if window_norm:
        spectrogram = spectrogram * window.pow(2.0).sum().sqrt()

    # pack batch
    shape = spectrogram.size()
    spectrogram = spectrogram.reshape(-1, shape[-2], shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    waveform = torch.istft(
        input=spectrogram,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        normalized=frame_length_norm,
        onesided=onesided,
        length=length + 2 * pad if length is not None else None,
        return_complex=False,
    )

    if length is not None and pad > 0:
        # remove padding from front and back
        waveform = waveform[:, pad:-pad]

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform


def _get_spec_norms(normalized: Union[str, bool]):
    frame_length_norm, window_norm = False, False
    if torch.jit.isinstance(normalized, str):
        if normalized not in ["frame_length", "window"]:
            raise ValueError("Invalid normalized parameter: {}".format(normalized))
        if normalized == "frame_length":
            frame_length_norm = True
        elif normalized == "window":
            window_norm = True
    elif torch.jit.isinstance(normalized, bool):
        if normalized:
            window_norm = True
    else:
        raise TypeError("Input type not supported")
    return frame_length_norm, window_norm


def _get_complex_dtype(real_dtype: torch.dtype):
    if real_dtype == torch.double:
        return torch.cdouble
    if real_dtype == torch.float:
        return torch.cfloat
    if real_dtype == torch.half:
        return torch.complex32
    raise ValueError(f"Unexpected dtype {real_dtype}")


def griffinlim(
    specgram: Tensor,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    power: float,
    n_iter: int,
    momentum: float,
    length: Optional[int],
    rand_init: bool,
) -> Tensor:
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Implementation ported from
    *librosa* :cite:`brian_mcfee-proc-scipy-2015`, *A fast Griffin-Lim algorithm* :cite:`6701851`
    and *Signal estimation from modified short-time Fourier transform* :cite:`1172092`.

    Args:
        specgram (Tensor): A magnitude-only STFT spectrogram of dimension `(..., freq, frames)`
            where freq is ``n_fft // 2 + 1``.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins
        hop_length (int): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        win_length (int): Window size. (Default: ``n_fft``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for magnitude, 2 for power, etc.
        n_iter (int): Number of iteration for phase recovery process.
        momentum (float): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge.
        length (int or None): Array length of the expected output.
        rand_init (bool): Initializes phase randomly if True, to zero otherwise.

    Returns:
        Tensor: waveform of `(..., time)`, where time equals the ``length`` parameter if given.
    """
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in range [0, 1). Found: {}".format(momentum))

    momentum = momentum / (1 + momentum)

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    specgram = specgram.pow(1 / power)

    # initialize the phase
    if rand_init:
        angles = torch.rand(specgram.size(), dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)
    else:
        angles = torch.full(specgram.size(), 1, dtype=_get_complex_dtype(specgram.dtype), device=specgram.device)

    # And initialize the previous iterate to 0
    tprev = torch.tensor(0.0, dtype=specgram.dtype, device=specgram.device)
    for _ in range(n_iter):
        # Invert with our current estimate of the phases
        inverse = torch.istft(
            specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
        )

        # Rebuild the spectrogram
        rebuilt = torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum)
        angles = angles.div(angles.abs().add(1e-16))

        # Store the previous iterate
        tprev = rebuilt

    # Return the final phase estimates
    waveform = torch.istft(
        specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
    )

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform


def amplitude_to_DB(
    x: Tensor, multiplier: float, amin: float, db_multiplier: float, top_db: Optional[float] = None
) -> Tensor:
    r"""Turn a spectrogram from the power/amplitude scale to the decibel scale.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The output of each tensor in a batch depends on the maximum value of that tensor,
    and so may return different values for an audio clip split into snippets vs. a full clip.

    Args:

        x (Tensor): Input spectrogram(s) before being converted to decibel scale.
            The expected shapes are ``(freq, time)``, ``(channel, freq, time)`` or
            ``(..., batch, channel, freq, time)``.

            .. note::

               When ``top_db`` is specified, cut-off values are computed for each audio
               in the batch. Therefore if the input shape is 4D (or larger), different
               cut-off values are used for audio data in the batch.
               If the input shape is 2D or 3D, a single cutoff value is used.

        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp ``x``
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (float or None, optional): Minimum negative cut-off in decibels. A reasonable number
            is 80. (Default: ``None``)

    Returns:
        Tensor: Output tensor in decibel scale
    """
    x_db = multiplier * torch.log10(torch.clamp(x, min=amin))
    x_db -= multiplier * db_multiplier

    if top_db is not None:
        # Expand batch
        shape = x_db.size()
        packed_channels = shape[-3] if x_db.dim() > 2 else 1
        x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])

        x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - top_db).view(-1, 1, 1, 1))

        # Repack batch
        x_db = x_db.reshape(shape)

    return x_db


def DB_to_amplitude(x: Tensor, ref: float, power: float) -> Tensor:
    r"""Turn a tensor from the decibel scale to the power/amplitude scale.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        x (Tensor): Input tensor before being converted to power/amplitude scale.
        ref (float): Reference which the output will be scaled by.
        power (float): If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.

    Returns:
        Tensor: Output tensor in power/amplitude scale.
    """
    return ref * torch.pow(torch.pow(10.0, 0.1 * x), power)


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels: Tensor, mel_scale: str = "htk") -> Tensor:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (Tensor): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """

    if mel_scale not in ["slaney", "htk"]:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def _create_triangular_filterbank(
    all_freqs: Tensor,
    f_pts: Tensor,
) -> Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb


def melscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
) -> Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/mel_fbanks.png
           :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (str or None, optional): If "slaney", divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A @ melscale_fbanks(A.size(-1), ...)``.

    """

    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def linear_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_filter: int,
    sample_rate: int,
) -> Tensor:
    r"""Creates a linear triangular filterbank.

    .. devices:: CPU

    .. properties:: TorchScript

    Note:
        For the sake of the numerical compatibility with librosa, not all the coefficients
        in the resulting filter bank has magnitude of 1.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/lin_fbanks.png
           :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_filter (int): Number of (linear) triangular filter
        sample_rate (int): Sample rate of the audio waveform

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_filter``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * linear_fbanks(A.size(-1), ...)``.
    """
    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # filter mid-points
    f_pts = torch.linspace(f_min, f_max, n_filter + 2)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    return fb


def create_dct(n_mfcc: int, n_mels: int, norm: Optional[str]) -> Tensor:
    r"""Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either "ortho" or None)

    Returns:
        Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """

    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)

    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def mu_law_encoding(x: Tensor, quantization_channels: int) -> Tensor:
    r"""Encode signal based on mu-law companding.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm expects the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1.

    Args:
        x (Tensor): Input tensor
        quantization_channels (int): Number of channels

    Returns:
        Tensor: Input after mu-law encoding
    """
    mu = quantization_channels - 1.0
    if not x.is_floating_point():
        warnings.warn(
            "The input Tensor must be of floating type. \
            This will be an error in the v0.12 release."
        )
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


def mu_law_decoding(x_mu: Tensor, quantization_channels: int) -> Tensor:
    r"""Decode mu-law encoded signal.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        x_mu (Tensor): Input tensor
        quantization_channels (int): Number of channels

    Returns:
        Tensor: Input after mu-law decoding
    """
    mu = quantization_channels - 1.0
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype)
    x = ((x_mu) / mu) * 2 - 1.0
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
    return x


def phase_vocoder(complex_specgrams: Tensor, rate: float, phase_advance: Tensor) -> Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a factor of ``rate``.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        complex_specgrams (Tensor):
            A tensor of dimension `(..., freq, num_frame)` with complex dtype.
        rate (float): Speed-up factor
        phase_advance (Tensor): Expected phase advance in each bin. Dimension of `(freq, 1)`

    Returns:
        Tensor:
            Stretched spectrogram. The resulting tensor is of the same dtype as the input
            spectrogram, but the number of frames is changed to ``ceil(num_frame / rate)``.

    Example
        >>> freq, hop_length = 1025, 512
        >>> # (channel, freq, time)
        >>> complex_specgrams = torch.randn(2, freq, 300, dtype=torch.cfloat)
        >>> rate = 1.3 # Speed up by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, freq)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([2, 1025, 231])
    """
    if rate == 1.0:
        return complex_specgrams

    # pack batch
    shape = complex_specgrams.size()
    complex_specgrams = complex_specgrams.reshape([-1] + list(shape[-2:]))

    # Figures out the corresponding real dtype, i.e. complex128 -> float64, complex64 -> float32
    # Note torch.real is a view so it does not incur any memory copy.
    real_dtype = torch.real(complex_specgrams).dtype
    time_steps = torch.arange(0, complex_specgrams.size(-1), rate, device=complex_specgrams.device, dtype=real_dtype)

    alphas = time_steps % 1.0
    phase_0 = complex_specgrams[..., :1].angle()

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 2])

    # (new_bins, freq, 2)
    complex_specgrams_0 = complex_specgrams.index_select(-1, time_steps.long())
    complex_specgrams_1 = complex_specgrams.index_select(-1, (time_steps + 1).long())

    angle_0 = complex_specgrams_0.angle()
    angle_1 = complex_specgrams_1.angle()

    norm_0 = complex_specgrams_0.abs()
    norm_1 = complex_specgrams_1.abs()

    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    complex_specgrams_stretch = torch.polar(mag, phase_acc)

    # unpack batch
    complex_specgrams_stretch = complex_specgrams_stretch.reshape(shape[:-2] + complex_specgrams_stretch.shape[1:])
    return complex_specgrams_stretch


def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    if p == 1.0:
        return mask_param
    else:
        return min(mask_param, int(axis_length * p))


def mask_along_axis_iid(
    specgrams: Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> Tensor:
    r"""Apply a mask along ``axis``.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Mask will be applied from indices ``[v_0, v_0 + v)``,
    where ``v`` is sampled from ``uniform(0, max_v)`` and
    ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``,
    with ``max_v = mask_param`` when ``p = 1.0`` and
    ``max_v = min(mask_param, floor(specgrams.size(axis) * p))`` otherwise.

    Args:
        specgrams (Tensor): Real spectrograms `(..., freq, time)`, with at least 3 dimensions.
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on, which should be the one of the last two dimensions.
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)

    Returns:
        Tensor: Masked spectrograms with the same dimensions as input specgrams Tensor`
    """

    dim = specgrams.dim()

    if dim < 3:
        raise ValueError(f"Spectrogram must have at least three dimensions ({dim} given).")

    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            f"Only Frequency and Time masking are supported (axis {dim-2} and axis {dim-1} supported; {axis} given)."
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgrams.shape[axis])
    if mask_param < 1:
        return specgrams

    device = specgrams.device
    dtype = specgrams.dtype

    value = torch.rand(specgrams.shape[: (dim - 2)], device=device, dtype=dtype) * mask_param
    min_value = torch.rand(specgrams.shape[: (dim - 2)], device=device, dtype=dtype) * (specgrams.size(axis) - value)

    # Create broadcastable mask
    mask_start = min_value.long()[..., None, None]
    mask_end = (min_value.long() + value.long())[..., None, None]
    mask = torch.arange(0, specgrams.size(axis), device=device, dtype=dtype)

    # Per batch example masking
    specgrams = specgrams.transpose(axis, -1)
    specgrams = specgrams.masked_fill((mask >= mask_start) & (mask < mask_end), mask_value)
    specgrams = specgrams.transpose(axis, -1)

    return specgrams


def mask_along_axis(
    specgram: Tensor,
    mask_param: int,
    mask_value: float,
    axis: int,
    p: float = 1.0,
) -> Tensor:
    r"""Apply a mask along ``axis``.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Mask will be applied from indices ``[v_0, v_0 + v)``,
    where ``v`` is sampled from ``uniform(0, max_v)`` and
    ``v_0`` from ``uniform(0, specgram.size(axis) - v)``, with
    ``max_v = mask_param`` when ``p = 1.0`` and
    ``max_v = min(mask_param, floor(specgram.size(axis) * p))``
    otherwise.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrograms `(..., freq, time)`, with at least 2 dimensions.
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on, which should be the one of the last two dimensions.
        p (float, optional): maximum proportion of columns that can be masked. (Default: 1.0)

    Returns:
        Tensor: Masked spectrograms with the same dimensions as input specgram Tensor
    """
    dim = specgram.dim()

    if dim < 2:
        raise ValueError(f"Spectrogram must have at least two dimensions (time and frequency) ({dim} given).")

    if axis not in [dim - 2, dim - 1]:
        raise ValueError(
            f"Only Frequency and Time masking are supported (axis {dim-2} and axis {dim-1} supported; {axis} given)."
        )

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    # After packing, specgram is a 3D tensor, and the axis corresponding to the to-be-masked dimension
    # is now (axis - dim + 3), e.g. a tensor of shape (10, 2, 50, 10, 2) becomes a tensor of shape (1000, 10, 2).
    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (specgram.size(axis - dim + 3) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    mask = torch.arange(0, specgram.shape[axis - dim + 3], device=specgram.device, dtype=specgram.dtype)
    mask = (mask >= mask_start) & (mask < mask_end)
    # unsqueeze the mask if the axis is frequency
    if axis == dim - 2:
        mask = mask.unsqueeze(-1)

    if mask_end - mask_start >= mask_param:
        raise ValueError("Number of columns to be masked should be less than mask_param")

    specgram = specgram.masked_fill(mask, mask_value)

    # unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram


def compute_deltas(specgram: Tensor, win_length: int = 5, mode: str = "replicate") -> Tensor:
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    .. math::
       d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension `(..., freq, time)`
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension `(..., freq, time)`

    Example
        >>> specgram = torch.randn(1, 40, 1000)
        >>> delta = compute_deltas(specgram)
        >>> delta2 = compute_deltas(delta)
    """
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    if win_length < 3:
        raise ValueError(f"Window length should be greater than or equal to 3. Found win_length {win_length}")

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(specgram.shape[1], 1, 1)

    output = torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom

    # unpack batch
    output = output.reshape(shape)

    return output


def _compute_nccf(waveform: Tensor, sample_rate: int, frame_time: float, freq_low: int) -> Tensor:
    r"""
    Compute Normalized Cross-Correlation Function (NCCF).

    .. math::
        \phi_i(m) = \frac{\sum_{n=b_i}^{b_i + N-1} w(n) w(m+n)}{\sqrt{E(b_i) E(m+b_i)}},

    where
    :math:`\phi_i(m)` is the NCCF at frame :math:`i` with lag :math:`m`,
    :math:`w` is the waveform,
    :math:`N` is the length of a frame,
    :math:`b_i` is the beginning of frame :math:`i`,
    :math:`E(j)` is the energy :math:`\sum_{n=j}^{j+N-1} w^2(n)`.
    """

    EPSILON = 10 ** (-9)

    # Number of lags to check
    lags = int(math.ceil(sample_rate / freq_low))

    frame_size = int(math.ceil(sample_rate * frame_time))

    waveform_length = waveform.size()[-1]
    num_of_frames = int(math.ceil(waveform_length / frame_size))

    p = lags + num_of_frames * frame_size - waveform_length
    waveform = torch.nn.functional.pad(waveform, (0, p))

    # Compute lags
    output_lag = []
    for lag in range(1, lags + 1):
        s1 = waveform[..., :-lag].unfold(-1, frame_size, frame_size)[..., :num_of_frames, :]
        s2 = waveform[..., lag:].unfold(-1, frame_size, frame_size)[..., :num_of_frames, :]

        output_frames = (
            (s1 * s2).sum(-1)
            / (EPSILON + torch.linalg.vector_norm(s1, ord=2, dim=-1)).pow(2)
            / (EPSILON + torch.linalg.vector_norm(s2, ord=2, dim=-1)).pow(2)
        )

        output_lag.append(output_frames.unsqueeze(-1))

    nccf = torch.cat(output_lag, -1)

    return nccf


def _combine_max(a: Tuple[Tensor, Tensor], b: Tuple[Tensor, Tensor], thresh: float = 0.99) -> Tuple[Tensor, Tensor]:
    """
    Take value from first if bigger than a multiplicative factor of the second, elementwise.
    """
    mask = a[0] > thresh * b[0]
    values = mask * a[0] + ~mask * b[0]
    indices = mask * a[1] + ~mask * b[1]
    return values, indices


def _find_max_per_frame(nccf: Tensor, sample_rate: int, freq_high: int) -> Tensor:
    r"""
    For each frame, take the highest value of NCCF,
    apply centered median smoothing, and convert to frequency.

    Note: If the max among all the lags is very close
    to the first half of lags, then the latter is taken.
    """

    lag_min = int(math.ceil(sample_rate / freq_high))

    # Find near enough max that is smallest

    best = torch.max(nccf[..., lag_min:], -1)

    half_size = nccf.shape[-1] // 2
    half = torch.max(nccf[..., lag_min:half_size], -1)

    best = _combine_max(half, best)
    indices = best[1]

    # Add back minimal lag
    indices += lag_min
    # Add 1 empirical calibration offset
    indices += 1

    return indices


def _median_smoothing(indices: Tensor, win_length: int) -> Tensor:
    r"""
    Apply median smoothing to the 1D tensor over the given window.
    """

    # Centered windowed
    pad_length = (win_length - 1) // 2

    # "replicate" padding in any dimension
    indices = torch.nn.functional.pad(indices, (pad_length, 0), mode="constant", value=0.0)

    indices[..., :pad_length] = torch.cat(pad_length * [indices[..., pad_length].unsqueeze(-1)], dim=-1)
    roll = indices.unfold(-1, win_length, 1)

    values, _ = torch.median(roll, -1)
    return values


def detect_pitch_frequency(
    waveform: Tensor,
    sample_rate: int,
    frame_time: float = 10 ** (-2),
    win_length: int = 30,
    freq_low: int = 85,
    freq_high: int = 3400,
) -> Tensor:
    r"""Detect pitch frequency.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    It is implemented using normalized cross-correlation function and median smoothing.

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., freq, time)`
        sample_rate (int): The sample rate of the waveform (Hz)
        frame_time (float, optional): Duration of a frame (Default: ``10 ** (-2)``).
        win_length (int, optional): The window length for median smoothing (in number of frames) (Default: ``30``).
        freq_low (int, optional): Lowest frequency that can be detected (Hz) (Default: ``85``).
        freq_high (int, optional): Highest frequency that can be detected (Hz) (Default: ``3400``).

    Returns:
        Tensor: Tensor of freq of dimension `(..., frame)`
    """
    # pack batch
    shape = list(waveform.size())
    waveform = waveform.reshape([-1] + shape[-1:])

    nccf = _compute_nccf(waveform, sample_rate, frame_time, freq_low)
    indices = _find_max_per_frame(nccf, sample_rate, freq_high)
    indices = _median_smoothing(indices, win_length)

    # Convert indices to frequency
    EPSILON = 10 ** (-9)
    freq = sample_rate / (EPSILON + indices.to(torch.float))

    # unpack batch
    freq = freq.reshape(shape[:-1] + list(freq.shape[-1:]))

    return freq


def sliding_window_cmn(
    specgram: Tensor,
    cmn_window: int = 600,
    min_cmn_window: int = 100,
    center: bool = False,
    norm_vars: bool = False,
) -> Tensor:
    r"""
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        specgram (Tensor): Tensor of spectrogram of dimension `(..., time, freq)`
        cmn_window (int, optional): Window in frames for running average CMN computation (int, default = 600)
        min_cmn_window (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center == false, ignored if center==true (int, default = 100)
        center (bool, optional): If true, use a window centered on the current frame
            (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
        norm_vars (bool, optional): If true, normalize variance to one. (bool, default = false)

    Returns:
        Tensor: Tensor matching input shape `(..., freq, time)`
    """
    input_shape = specgram.shape
    num_frames, num_feats = input_shape[-2:]
    specgram = specgram.view(-1, num_frames, num_feats)
    num_channels = specgram.shape[0]

    dtype = specgram.dtype
    device = specgram.device
    last_window_start = last_window_end = -1
    cur_sum = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
    cur_sumsq = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
    cmn_specgram = torch.zeros(num_channels, num_frames, num_feats, dtype=dtype, device=device)
    for t in range(num_frames):
        window_start = 0
        window_end = 0
        if center:
            window_start = t - cmn_window // 2
            window_end = window_start + cmn_window
        else:
            window_start = t - cmn_window
            window_end = t + 1
        if window_start < 0:
            window_end -= window_start
            window_start = 0
        if not center:
            if window_end > t:
                window_end = max(t + 1, min_cmn_window)
        if window_end > num_frames:
            window_start -= window_end - num_frames
            window_end = num_frames
            if window_start < 0:
                window_start = 0
        if last_window_start == -1:
            input_part = specgram[:, window_start : window_end - window_start, :]
            cur_sum += torch.sum(input_part, 1)
            if norm_vars:
                cur_sumsq += torch.cumsum(input_part**2, 1)[:, -1, :]
        else:
            if window_start > last_window_start:
                frame_to_remove = specgram[:, last_window_start, :]
                cur_sum -= frame_to_remove
                if norm_vars:
                    cur_sumsq -= frame_to_remove**2
            if window_end > last_window_end:
                frame_to_add = specgram[:, last_window_end, :]
                cur_sum += frame_to_add
                if norm_vars:
                    cur_sumsq += frame_to_add**2
        window_frames = window_end - window_start
        last_window_start = window_start
        last_window_end = window_end
        cmn_specgram[:, t, :] = specgram[:, t, :] - cur_sum / window_frames
        if norm_vars:
            if window_frames == 1:
                cmn_specgram[:, t, :] = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
            else:
                variance = cur_sumsq
                variance = variance / window_frames
                variance -= (cur_sum**2) / (window_frames**2)
                variance = torch.pow(variance, -0.5)
                cmn_specgram[:, t, :] *= variance

    cmn_specgram = cmn_specgram.view(input_shape[:-2] + (num_frames, num_feats))
    if len(input_shape) == 2:
        cmn_specgram = cmn_specgram.squeeze(0)
    return cmn_specgram


def spectral_centroid(
    waveform: Tensor,
    sample_rate: int,
    pad: int,
    window: Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> Tensor:
    r"""Compute the spectral centroid for each channel along the time axis.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The spectral centroid is defined as the weighted average of the
    frequency values, weighted by their magnitude.

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., time)`
        sample_rate (int): Sample rate of the audio waveform
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size

    Returns:
        Tensor: Dimension `(..., time)`
    """
    specgram = spectrogram(
        waveform,
        pad=pad,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,
        normalized=False,
    )
    freqs = torch.linspace(0, sample_rate // 2, steps=1 + n_fft // 2, device=specgram.device).reshape((-1, 1))
    freq_dim = -2
    return (freqs * specgram).sum(dim=freq_dim) / specgram.sum(dim=freq_dim)


@torchaudio._extension.fail_if_no_sox
@deprecated("Please migrate to :py:class:`torchaudio.io.AudioEffector`.", remove=False)
def apply_codec(
    waveform: Tensor,
    sample_rate: int,
    format: str,
    channels_first: bool = True,
    compression: Optional[float] = None,
    encoding: Optional[str] = None,
    bits_per_sample: Optional[int] = None,
) -> Tensor:
    r"""
    Apply codecs as a form of augmentation.

    .. devices:: CPU

    Args:
        waveform (Tensor): Audio data. Must be 2 dimensional. See also ```channels_first```.
        sample_rate (int): Sample rate of the audio waveform.
        format (str): File format.
        channels_first (bool, optional):
            When True, both the input and output Tensor have dimension `(channel, time)`.
            Otherwise, they have dimension `(time, channel)`.
        compression (float or None, optional): Used for formats other than WAV.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.
        encoding (str or None, optional): Changes the encoding for the supported formats.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.
        bits_per_sample (int or None, optional): Changes the bit depth for the supported formats.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.

    Returns:
        Tensor: Resulting Tensor.
        If ``channels_first=True``, it has `(channel, time)` else `(time, channel)`.
    """
    with tempfile.NamedTemporaryFile() as f:
        torchaudio.backend.sox_io_backend.save(
            f.name, waveform, sample_rate, channels_first, compression, format, encoding, bits_per_sample
        )
        augmented, sr = torchaudio.backend.sox_io_backend.load(f.name, channels_first=channels_first, format=format)
    if sr != sample_rate:
        augmented = resample(augmented, sr, sample_rate)
    return augmented


_CPU = torch.device("cpu")


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    device: torch.device = _CPU,
    dtype: Optional[torch.dtype] = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. "
            "To work around this, manually convert both frequencies to integer values "
            "that maintain their resampling rate ratio before passing them into the function. "
            "Example: To downsample a 44100 hz waveform by a factor of 8, use "
            "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
            "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
        )

    if resampling_method in ["sinc_interpolation", "kaiser_window"]:
        method_map = {
            "sinc_interpolation": "sinc_interp_hann",
            "kaiser_window": "sinc_interp_kaiser",
        }
        warnings.warn(
            f'"{resampling_method}" resampling method name is being deprecated and replaced by '
            f'"{method_map[resampling_method]}" in the next release. '
            "The default behavior remains unchanged."
        )
    elif resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    base_freq *= rolloff

    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else torch.float64

    idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq

    t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interp_hann":
        window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = torch.tensor(float(beta))
        window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = kernels.to(dtype=torch.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: Tensor,
    width: int,
):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.view(shape[:-1] + resampled.shape[-1:])
    return resampled


def resample(
    waveform: Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
) -> Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation. :cite:`RESAMPLE`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        ``transforms.Resample`` precomputes and reuses the resampling kernel, so using it will result in
        more efficient computation if resampling multiple waveforms with the same resampling parameters.

    Args:
        waveform (Tensor): The input signal of dimension `(..., time)`
        orig_freq (int): The original frequency of the signal
        new_freq (int): The desired frequency
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``"sinc_interp_hann"``, ``"sinc_interp_kaiser"``] (Default: ``"sinc_interp_hann"``)
        beta (float or None, optional): The shape parameter used for kaiser window.

    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """

    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequecy should be positive")

    if orig_freq == new_freq:
        return waveform

    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.device,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled


@torch.jit.unused
def edit_distance(seq1: Sequence, seq2: Sequence) -> int:
    """
    Calculate the word level edit (Levenshtein) distance between two sequences.

    .. devices:: CPU

    The function computes an edit distance allowing deletion, insertion and
    substitution. The result is an integer.

    For most applications, the two input sequences should be the same type. If
    two strings are given, the output is the edit distance between the two
    strings (character edit distance). If two lists of strings are given, the
    output is the edit distance between sentences (word edit distance). Users
    may want to normalize the output by the length of the reference sequence.

    Args:
        seq1 (Sequence): the first sequence to compare.
        seq2 (Sequence): the second sequence to compare.
    Returns:
        int: The distance between the first and second sequences.
    """
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return int(dold[-1])


def loudness(waveform: Tensor, sample_rate: int):
    r"""Measure audio loudness according to the ITU-R BS.1770-4 recommendation.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`
        sample_rate (int): sampling rate of the waveform

    Returns:
        Tensor: loudness estimates (LKFS)

    Reference:
        - https://www.itu.int/rec/R-REC-BS.1770-4-201510-I/en
    """

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")

    gate_duration = 0.4
    overlap = 0.75
    gamma_abs = -70.0
    kweight_bias = -0.691
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    loudness = -0.691 + 10 * torch.log10(energy_weighted)

    # Apply absolute gating of the blocks
    gated_blocks = loudness > gamma_abs
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    gamma_rel = kweight_bias + 10 * torch.log10(energy_weighted) - 10

    # Apply relative gating of the blocks
    gated_blocks = torch.logical_and(gated_blocks.squeeze(-2), loudness > gamma_rel.unsqueeze(-1))
    gated_blocks = gated_blocks.unsqueeze(-2)

    energy_filtered = torch.sum(gated_blocks * energy, dim=-1) / torch.count_nonzero(gated_blocks, dim=-1)
    energy_weighted = torch.sum(g * energy_filtered, dim=-1)
    LKFS = kweight_bias + 10 * torch.log10(energy_weighted)
    return LKFS


def pitch_shift(
    waveform: Tensor,
    sample_rate: int,
    n_steps: int,
    bins_per_octave: int = 12,
    n_fft: int = 512,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    window: Optional[Tensor] = None,
) -> Tensor:
    """
    Shift the pitch of a waveform by ``n_steps`` steps.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        waveform (Tensor): The input waveform of shape `(..., time)`.
        sample_rate (int): Sample rate of `waveform`.
        n_steps (int): The (fractional) steps to shift `waveform`.
        bins_per_octave (int, optional): The number of steps per octave (Default: ``12``).
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins (Default: ``512``).
        win_length (int or None, optional): Window size. If None, then ``n_fft`` is used. (Default: ``None``).
        hop_length (int or None, optional): Length of hop between STFT windows. If None, then
            ``win_length // 4`` is used (Default: ``None``).
        window (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window.
            If None, then ``torch.hann_window(win_length)`` is used (Default: ``None``).


    Returns:
        Tensor: The pitch-shifted audio waveform of shape `(..., time)`.
    """
    waveform_stretch = _stretch_waveform(
        waveform,
        n_steps,
        bins_per_octave,
        n_fft,
        win_length,
        hop_length,
        window,
    )
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    waveform_shift = resample(waveform_stretch, int(sample_rate / rate), sample_rate)

    return _fix_waveform_shape(waveform_shift, waveform.size())


def _stretch_waveform(
    waveform: Tensor,
    n_steps: int,
    bins_per_octave: int = 12,
    n_fft: int = 512,
    win_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    window: Optional[Tensor] = None,
) -> Tensor:
    """
    Pitch shift helper function to preprocess and stretch waveform before resampling step.

    Args:
        See pitch_shift arg descriptions.

    Returns:
        Tensor: The preprocessed waveform stretched prior to resampling.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(window_length=win_length, device=waveform.device)

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    ori_len = shape[-1]
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    phase_advance = torch.linspace(0, math.pi * hop_length, spec_f.shape[-2], device=spec_f.device)[..., None]
    spec_stretch = phase_vocoder(spec_f, rate, phase_advance)
    len_stretch = int(round(ori_len / rate))
    waveform_stretch = torch.istft(
        spec_stretch, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=len_stretch
    )
    return waveform_stretch


def _fix_waveform_shape(
    waveform_shift: Tensor,
    shape: List[int],
) -> Tensor:
    """
    PitchShift helper function to process after resampling step to fix the shape back.

    Args:
        waveform_shift(Tensor): The waveform after stretch and resample
        shape (List[int]): The shape of initial waveform

    Returns:
        Tensor: The pitch-shifted audio waveform of shape `(..., time)`.
    """
    ori_len = shape[-1]
    shift_len = waveform_shift.size()[-1]
    if shift_len > ori_len:
        waveform_shift = waveform_shift[..., :ori_len]
    else:
        waveform_shift = torch.nn.functional.pad(waveform_shift, [0, ori_len - shift_len])

    # unpack batch
    waveform_shift = waveform_shift.view(shape[:-1] + waveform_shift.shape[-1:])
    return waveform_shift


def rnnt_loss(
    logits: Tensor,
    targets: Tensor,
    logit_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = -1,
    clamp: float = -1,
    reduction: str = "mean",
    fused_log_softmax: bool = True,
):
    """Compute the RNN Transducer loss from *Sequence Transduction with Recurrent Neural Networks*
    :cite:`graves2012sequence`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The RNN Transducer loss extends the CTC loss by defining a distribution over output
    sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        logits (Tensor): Tensor of dimension `(batch, max seq length, max target length + 1, class)`
            containing output from joiner
        targets (Tensor): Tensor of dimension `(batch, max target length)` containing targets with zero padded
        logit_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of each sequence from encoder
        target_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of targets for each sequence
        blank (int, optional): blank label (Default: ``-1``)
        clamp (float, optional): clamp for gradients (Default: ``-1``)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. (Default: ``"mean"``)
        fused_log_softmax (bool): set to False if calling log_softmax outside of loss (Default: ``True``)
    Returns:
        Tensor: Loss with the reduction option applied. If ``reduction`` is  ``"none"``, then size `(batch)`,
        otherwise scalar.
    """
    if reduction not in ["none", "mean", "sum"]:
        raise ValueError('reduction should be one of "none", "mean", or "sum"')

    if blank < 0:  # reinterpret blank index if blank < 0.
        blank = logits.shape[-1] + blank

    costs, _ = torch.ops.torchaudio.rnnt_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=blank,
        clamp=clamp,
        fused_log_softmax=fused_log_softmax,
    )

    if reduction == "mean":
        return costs.mean()
    elif reduction == "sum":
        return costs.sum()

    return costs


def psd(
    specgram: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
    eps: float = 1e-10,
) -> Tensor:
    """Compute cross-channel power spectral density (PSD) matrix.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        specgram (torch.Tensor): Multi-channel complex-valued spectrum.
            Tensor with dimensions `(..., channel, freq, time)`.
        mask (torch.Tensor or None, optional): Time-Frequency mask for normalization.
            Tensor with dimensions `(..., freq, time)`. (Default: ``None``)
        normalize (bool, optional): If ``True``, normalize the mask along the time dimension. (Default: ``True``)
        eps (float, optional): Value to add to the denominator in mask normalization. (Default: ``1e-15``)

    Returns:
        torch.Tensor: The complex-valued PSD matrix of the input spectrum.
        Tensor with dimensions `(..., freq, channel, channel)`
    """
    specgram = specgram.transpose(-3, -2)  # shape (freq, channel, time)
    # outer product:
    # (..., ch_1, time) x (..., ch_2, time) -> (..., time, ch_1, ch_2)
    psd = torch.einsum("...ct,...et->...tce", [specgram, specgram.conj()])

    if mask is not None:
        if mask.shape[:-1] != specgram.shape[:-2] or mask.shape[-1] != specgram.shape[-1]:
            raise ValueError(
                "The dimensions of mask except the channel dimension should be the same as specgram."
                f"Found {mask.shape} for mask and {specgram.shape} for specgram."
            )
        # Normalized mask along time dimension:
        if normalize:
            mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

        psd = psd * mask[..., None, None]

    psd = psd.sum(dim=-3)
    return psd


def _compute_mat_trace(input: torch.Tensor, dim1: int = -1, dim2: int = -2) -> torch.Tensor:
    r"""Compute the trace of a Tensor along ``dim1`` and ``dim2`` dimensions.

    Args:
        input (torch.Tensor): Tensor with dimensions `(..., channel, channel)`.
        dim1 (int, optional): The first dimension of the diagonal matrix.
            (Default: ``-1``)
        dim2 (int, optional): The second dimension of the diagonal matrix.
            (Default: ``-2``)

    Returns:
        Tensor: The trace of the input Tensor.
    """
    if input.ndim < 2:
        raise ValueError("The dimension of the tensor must be at least 2.")
    if input.shape[dim1] != input.shape[dim2]:
        raise ValueError("The size of ``dim1`` and ``dim2`` must be the same.")
    input = torch.diagonal(input, 0, dim1=dim1, dim2=dim2)
    return input.sum(dim=-1)


def _tik_reg(mat: torch.Tensor, reg: float = 1e-7, eps: float = 1e-8) -> torch.Tensor:
    """Perform Tikhonov regularization (only modifying real part).

    Args:
        mat (torch.Tensor): Input matrix with dimensions `(..., channel, channel)`.
        reg (float, optional): Regularization factor. (Default: 1e-8)
        eps (float, optional): Value to avoid the correlation matrix is all-zero. (Default: ``1e-8``)

    Returns:
        Tensor: Regularized matrix with dimensions `(..., channel, channel)`.
    """
    # Add eps
    C = mat.size(-1)
    eye = torch.eye(C, dtype=mat.dtype, device=mat.device)
    epsilon = _compute_mat_trace(mat).real[..., None, None] * reg
    # in case that correlation_matrix is all-zero
    epsilon = epsilon + eps
    mat = mat + epsilon * eye[..., :, :]
    return mat


def _assert_psd_matrices(psd_s: torch.Tensor, psd_n: torch.Tensor) -> None:
    """Assertion checks of the PSD matrices of target speech and noise.

    Args:
        psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
    """
    if psd_s.ndim < 3 or psd_n.ndim < 3:
        raise ValueError(
            "Expected at least 3D Tensor (..., freq, channel, channel) for psd_s and psd_n. "
            f"Found {psd_s.shape} for psd_s and {psd_n.shape} for psd_n."
        )
    if not (psd_s.is_complex() and psd_n.is_complex()):
        raise TypeError(
            "The type of psd_s and psd_n must be ``torch.cfloat`` or ``torch.cdouble``. "
            f"Found {psd_s.dtype} for psd_s and {psd_n.dtype} for psd_n."
        )
    if psd_s.shape != psd_n.shape:
        raise ValueError(
            f"The dimensions of psd_s and psd_n should be the same. Found {psd_s.shape} and {psd_n.shape}."
        )
    if psd_s.shape[-1] != psd_s.shape[-2]:
        raise ValueError(f"The last two dimensions of psd_s should be the same. Found {psd_s.shape}.")


def mvdr_weights_souden(
    psd_s: Tensor,
    psd_n: Tensor,
    reference_channel: Union[int, Tensor],
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Tensor:
    r"""Compute the Minimum Variance Distortionless Response (*MVDR* :cite:`capon1969high`) beamforming weights
    by the method proposed by *Souden et, al.* :cite:`souden2009optimal`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the power spectral density (PSD) matrix of target speech :math:`\bf{\Phi}_{\textbf{SS}}`,
    the PSD matrix of noise :math:`\bf{\Phi}_{\textbf{NN}}`, and a one-hot vector that represents the
    reference channel :math:`\bf{u}`, the method computes the MVDR beamforming weight martrix
    :math:`\textbf{w}_{\text{MVDR}}`. The formula is defined as:

    .. math::
        \textbf{w}_{\text{MVDR}}(f) =
        \frac{{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bf{\Phi}_{\textbf{SS}}}}(f)}
        {\text{Trace}({{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f) \bf{\Phi}_{\textbf{SS}}}(f))}}\bm{u}

    Args:
        psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
        reference_channel (int or torch.Tensor): Specifies the reference channel.
            If the dtype is ``int``, it represents the reference channel index.
            If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
            is one-hot.
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
            (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
        eps (float, optional): Value to add to the denominator in the beamforming weight formula.
            (Default: ``1e-8``)

    Returns:
        torch.Tensor: The complex-valued MVDR beamforming weight matrix with dimensions `(..., freq, channel)`.
    """
    _assert_psd_matrices(psd_s, psd_n)

    if diagonal_loading:
        psd_n = _tik_reg(psd_n, reg=diag_eps)
    numerator = torch.linalg.solve(psd_n, psd_s)  # psd_n.inv() @ psd_s
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (_compute_mat_trace(numerator)[..., None, None] + eps)
    if torch.jit.isinstance(reference_channel, int):
        beamform_weights = ws[..., :, reference_channel]
    elif torch.jit.isinstance(reference_channel, Tensor):
        reference_channel = reference_channel.to(psd_n.dtype)
        # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
        beamform_weights = torch.einsum("...c,...c->...", [ws, reference_channel[..., None, None, :]])
    else:
        raise TypeError(f'Expected "int" or "Tensor" for reference_channel. Found: {type(reference_channel)}.')

    return beamform_weights


def mvdr_weights_rtf(
    rtf: Tensor,
    psd_n: Tensor,
    reference_channel: Optional[Union[int, Tensor]] = None,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> Tensor:
    r"""Compute the Minimum Variance Distortionless Response (*MVDR* :cite:`capon1969high`) beamforming weights
    based on the relative transfer function (RTF) and power spectral density (PSD) matrix of noise.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the relative transfer function (RTF) matrix or the steering vector of target speech :math:`\bm{v}`,
    the PSD matrix of noise :math:`\bf{\Phi}_{\textbf{NN}}`, and a one-hot vector that represents the
    reference channel :math:`\bf{u}`, the method computes the MVDR beamforming weight martrix
    :math:`\textbf{w}_{\text{MVDR}}`. The formula is defined as:

    .. math::
        \textbf{w}_{\text{MVDR}}(f) =
        \frac{{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bm{v}}(f)}}
        {{\bm{v}^{\mathsf{H}}}(f){\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bm{v}}(f)}

    where :math:`(.)^{\mathsf{H}}` denotes the Hermitian Conjugate operation.

    Args:
        rtf (torch.Tensor): The complex-valued RTF vector of target speech.
            Tensor with dimensions `(..., freq, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
        reference_channel (int or torch.Tensor): Specifies the reference channel.
            If the dtype is ``int``, it represents the reference channel index.
            If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
            is one-hot.
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
            (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
        eps (float, optional): Value to add to the denominator in the beamforming weight formula.
            (Default: ``1e-8``)

    Returns:
        torch.Tensor: The complex-valued MVDR beamforming weight matrix with dimensions `(..., freq, channel)`.
    """
    if rtf.ndim < 2:
        raise ValueError(f"Expected at least 2D Tensor (..., freq, channel) for rtf. Found {rtf.shape}.")
    if psd_n.ndim < 3:
        raise ValueError(f"Expected at least 3D Tensor (..., freq, channel, channel) for psd_n. Found {psd_n.shape}.")
    if not (rtf.is_complex() and psd_n.is_complex()):
        raise TypeError(
            "The type of rtf and psd_n must be ``torch.cfloat`` or ``torch.cdouble``. "
            f"Found {rtf.dtype} for rtf and {psd_n.dtype} for psd_n."
        )
    if rtf.shape != psd_n.shape[:-1]:
        raise ValueError(
            "The dimensions of rtf and the dimensions withou the last dimension of psd_n should be the same. "
            f"Found {rtf.shape} for rtf and {psd_n.shape} for psd_n."
        )
    if psd_n.shape[-1] != psd_n.shape[-2]:
        raise ValueError(f"The last two dimensions of psd_n should be the same. Found {psd_n.shape}.")

    if diagonal_loading:
        psd_n = _tik_reg(psd_n, reg=diag_eps)
    # numerator = psd_n.inv() @ stv
    numerator = torch.linalg.solve(psd_n, rtf.unsqueeze(-1)).squeeze(-1)  # (..., freq, channel)
    # denominator = stv^H @ psd_n.inv() @ stv
    denominator = torch.einsum("...d,...d->...", [rtf.conj(), numerator])
    beamform_weights = numerator / (denominator.real.unsqueeze(-1) + eps)
    # normalize the numerator
    if reference_channel is not None:
        if torch.jit.isinstance(reference_channel, int):
            scale = rtf[..., reference_channel].conj()
        elif torch.jit.isinstance(reference_channel, Tensor):
            reference_channel = reference_channel.to(psd_n.dtype)
            scale = torch.einsum("...c,...c->...", [rtf.conj(), reference_channel[..., None, :]])
        else:
            raise TypeError(f'Expected "int" or "Tensor" for reference_channel. Found: {type(reference_channel)}.')

        beamform_weights = beamform_weights * scale[..., None]

    return beamform_weights


def rtf_evd(psd_s: Tensor) -> Tensor:
    r"""Estimate the relative transfer function (RTF) or the steering vector by eigenvalue decomposition.

    .. devices:: CPU CUDA

    .. properties:: TorchScript

    Args:
        psd_s (Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor of dimension `(..., freq, channel, channel)`

    Returns:
        Tensor: The estimated complex-valued RTF of target speech.
        Tensor of dimension `(..., freq, channel)`
    """
    if not psd_s.is_complex():
        raise TypeError(f"The type of psd_s must be ``torch.cfloat`` or ``torch.cdouble``. Found {psd_s.dtype}.")
    if psd_s.shape[-1] != psd_s.shape[-2]:
        raise ValueError(f"The last two dimensions of psd_s should be the same. Found {psd_s.shape}.")
    _, v = torch.linalg.eigh(psd_s)  # v is sorted along with eigenvalues in ascending order
    rtf = v[..., -1]  # choose the eigenvector with max eigenvalue
    return rtf


def rtf_power(
    psd_s: Tensor,
    psd_n: Tensor,
    reference_channel: Union[int, Tensor],
    n_iter: int = 3,
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
) -> Tensor:
    r"""Estimate the relative transfer function (RTF) or the steering vector by the power method.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        psd_s (torch.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (torch.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
        reference_channel (int or torch.Tensor): Specifies the reference channel.
            If the dtype is ``int``, it represents the reference channel index.
            If the dtype is ``torch.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
            is one-hot.
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
            (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)

    Returns:
        torch.Tensor: The estimated complex-valued RTF of target speech.
        Tensor of dimension `(..., freq, channel)`.
    """
    _assert_psd_matrices(psd_s, psd_n)
    if n_iter <= 0:
        raise ValueError("The number of iteration must be greater than 0.")

    # Apply diagonal loading to psd_n to improve robustness.
    if diagonal_loading:
        psd_n = _tik_reg(psd_n, reg=diag_eps)
    # phi is regarded as the first iteration
    phi = torch.linalg.solve(psd_n, psd_s)  # psd_n.inv() @ psd_s
    if torch.jit.isinstance(reference_channel, int):
        rtf = phi[..., reference_channel]
    elif torch.jit.isinstance(reference_channel, Tensor):
        reference_channel = reference_channel.to(psd_n.dtype)
        rtf = torch.einsum("...c,...c->...", [phi, reference_channel[..., None, None, :]])
    else:
        raise TypeError(f'Expected "int" or "Tensor" for reference_channel. Found: {type(reference_channel)}.')
    rtf = rtf.unsqueeze(-1)  # (..., freq, channel, 1)
    if n_iter >= 2:
        # The number of iterations in the for loop is `n_iter - 2`
        # because the `phi` above and `torch.matmul(psd_s, rtf)` are regarded as
        # two iterations.
        for _ in range(n_iter - 2):
            rtf = torch.matmul(phi, rtf)
        rtf = torch.matmul(psd_s, rtf)
    else:
        # if there is only one iteration, the rtf is the psd_s[..., referenc_channel]
        # which is psd_n @ phi @ ref_channel
        rtf = torch.matmul(psd_n, rtf)
    return rtf.squeeze(-1)


def apply_beamforming(beamform_weights: Tensor, specgram: Tensor) -> Tensor:
    r"""Apply the beamforming weight to the multi-channel noisy spectrum to obtain the single-channel enhanced spectrum.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    .. math::
        \hat{\textbf{S}}(f) = \textbf{w}_{\text{bf}}(f)^{\mathsf{H}} \textbf{Y}(f)

    where :math:`\textbf{w}_{\text{bf}}(f)` is the beamforming weight for the :math:`f`-th frequency bin,
    :math:`\textbf{Y}` is the multi-channel spectrum for the :math:`f`-th frequency bin.

    Args:
        beamform_weights (Tensor): The complex-valued beamforming weight matrix.
            Tensor of dimension `(..., freq, channel)`
        specgram (Tensor): The multi-channel complex-valued noisy spectrum.
            Tensor of dimension `(..., channel, freq, time)`

    Returns:
        Tensor: The single-channel complex-valued enhanced spectrum.
            Tensor of dimension `(..., freq, time)`
    """
    if beamform_weights.shape[:-2] != specgram.shape[:-3]:
        raise ValueError(
            "The dimensions except the last two dimensions of beamform_weights should be the same "
            "as the dimensions except the last three dimensions of specgram. "
            f"Found {beamform_weights.shape} for beamform_weights and {specgram.shape} for specgram."
        )

    if not (beamform_weights.is_complex() and specgram.is_complex()):
        raise TypeError(
            "The type of beamform_weights and specgram must be ``torch.cfloat`` or ``torch.cdouble``. "
            f"Found {beamform_weights.dtype} for beamform_weights and {specgram.dtype} for specgram."
        )

    # (..., freq, channel) x (..., channel, freq, time) -> (..., freq, time)
    specgram_enhanced = torch.einsum("...fc,...cft->...ft", [beamform_weights.conj(), specgram])
    return specgram_enhanced


def _check_shape_compatible(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.ndim != y.ndim:
        raise ValueError(f"The operands must be the same dimension (got {x.ndim} and {y.ndim}).")

    for i in range(x.ndim - 1):
        xi = x.size(i)
        yi = y.size(i)
        if xi == yi or xi == 1 or yi == 1:
            continue
        raise ValueError(f"Leading dimensions of x and y are not broadcastable (got {x.shape} and {y.shape}).")


def _check_convolve_mode(mode: str) -> None:
    valid_convolve_modes = ["full", "valid", "same"]
    if mode not in valid_convolve_modes:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")


def _apply_convolve_mode(conv_result: torch.Tensor, x_length: int, y_length: int, mode: str) -> torch.Tensor:
    valid_convolve_modes = ["full", "valid", "same"]
    if mode == "full":
        return conv_result
    elif mode == "valid":
        target_length = max(x_length, y_length) - min(x_length, y_length) + 1
        start_idx = (conv_result.size(-1) - target_length) // 2
        return conv_result[..., start_idx : start_idx + target_length]
    elif mode == "same":
        start_idx = (conv_result.size(-1) - x_length) // 2
        return conv_result[..., start_idx : start_idx + x_length]
    else:
        raise ValueError(f"Unrecognized mode value '{mode}'. Please specify one of {valid_convolve_modes}.")


def fftconvolve(x: torch.Tensor, y: torch.Tensor, mode: str = "full") -> torch.Tensor:
    r"""
    Convolves inputs along their last dimension using FFT. For inputs with large last dimensions, this function
    is generally much faster than :meth:`convolve`.
    Note that, in contrast to :meth:`torch.nn.functional.conv1d`, which actually applies the valid cross-correlation
    operator, this function applies the true `convolution`_ operator.
    Also note that this function can only output float tensors (int tensor inputs will be cast to float).

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        x (torch.Tensor): First convolution operand, with shape `(..., N)`.
        y (torch.Tensor): Second convolution operand, with shape `(..., M)`
            (leading dimensions must be broadcast-able with those of ``x``).
        mode (str, optional): Must be one of ("full", "valid", "same").

            * "full": Returns the full convolution result, with shape `(..., N + M - 1)`. (Default)
            * "valid": Returns the segment of the full convolution result corresponding to where
              the two inputs overlap completely, with shape `(..., max(N, M) - min(N, M) + 1)`.
            * "same": Returns the center segment of the full convolution result, with shape `(..., N)`.

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., L)`, where
        the leading dimensions match those of ``x`` and `L` is dictated by ``mode``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """
    _check_shape_compatible(x, y)
    _check_convolve_mode(mode)

    n = x.size(-1) + y.size(-1) - 1
    fresult = torch.fft.rfft(x, n=n) * torch.fft.rfft(y, n=n)
    result = torch.fft.irfft(fresult, n=n)
    return _apply_convolve_mode(result, x.size(-1), y.size(-1), mode)


def convolve(x: torch.Tensor, y: torch.Tensor, mode: str = "full") -> torch.Tensor:
    r"""
    Convolves inputs along their last dimension using the direct method.
    Note that, in contrast to :meth:`torch.nn.functional.conv1d`, which actually applies the valid cross-correlation
    operator, this function applies the true `convolution`_ operator.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        x (torch.Tensor): First convolution operand, with shape `(..., N)`.
        y (torch.Tensor): Second convolution operand, with shape `(..., M)`
            (leading dimensions must be broadcast-able with those of ``x``).
        mode (str, optional): Must be one of ("full", "valid", "same").

            * "full": Returns the full convolution result, with shape `(..., N + M - 1)`. (Default)
            * "valid": Returns the segment of the full convolution result corresponding to where
              the two inputs overlap completely, with shape `(..., max(N, M) - min(N, M) + 1)`.
            * "same": Returns the center segment of the full convolution result, with shape `(..., N)`.

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., L)`, where
        the leading dimensions match those of ``x`` and `L` is dictated by ``mode``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """
    _check_shape_compatible(x, y)
    _check_convolve_mode(mode)

    x_size, y_size = x.size(-1), y.size(-1)

    if x.size(-1) < y.size(-1):
        x, y = y, x

    if x.shape[:-1] != y.shape[:-1]:
        new_shape = [max(i, j) for i, j in zip(x.shape[:-1], y.shape[:-1])]
        x = x.broadcast_to(new_shape + [x.shape[-1]])
        y = y.broadcast_to(new_shape + [y.shape[-1]])

    num_signals = torch.tensor(x.shape[:-1]).prod()
    reshaped_x = x.reshape((int(num_signals), x.size(-1)))
    reshaped_y = y.reshape((int(num_signals), y.size(-1)))
    output = torch.nn.functional.conv1d(
        input=reshaped_x,
        weight=reshaped_y.flip(-1).unsqueeze(1),
        stride=1,
        groups=reshaped_x.size(0),
        padding=reshaped_y.size(-1) - 1,
    )
    output_shape = x.shape[:-1] + (-1,)
    result = output.reshape(output_shape)
    return _apply_convolve_mode(result, x_size, y_size, mode)


def add_noise(
    waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    """

    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    # compute scale
    if lengths is not None:
        mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
            -1
        )  # (*, L) < (*, 1) = (*, L)
        masked_waveform = waveform * mask
        masked_noise = noise * mask
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)


def speed(
    waveform: torch.Tensor, orig_freq: int, factor: float, lengths: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Adjusts waveform speed.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input signals, with shape `(..., time)`.
        orig_freq (int): Original frequency of the signals in ``waveform``.
        factor (float): Factor by which to adjust speed of input. Values greater than 1.0
            compress ``waveform`` in time, whereas values less than 1.0 stretch ``waveform`` in time.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform``, with shape `(...)`.
            If ``None``, all elements in ``waveform`` are treated as valid. (Default: ``None``)

    Returns:
        (torch.Tensor, torch.Tensor or None):
            torch.Tensor
                Speed-adjusted waveform, with shape `(..., new_time).`
            torch.Tensor or None
                If ``lengths`` is not ``None``, valid lengths of signals in speed-adjusted waveform,
                with shape `(...)`; otherwise, ``None``.
    """

    source_sample_rate = int(factor * orig_freq)
    target_sample_rate = int(orig_freq)

    gcd = math.gcd(source_sample_rate, target_sample_rate)
    source_sample_rate = source_sample_rate // gcd
    target_sample_rate = target_sample_rate // gcd

    if lengths is None:
        out_lengths = None
    else:
        out_lengths = torch.ceil(lengths * target_sample_rate / source_sample_rate).to(lengths.dtype)

    return resample(waveform, source_sample_rate, target_sample_rate), out_lengths


def preemphasis(waveform, coeff: float = 0.97) -> torch.Tensor:
    r"""Pre-emphasizes a waveform along its last dimension, i.e.
    for each signal :math:`x` in ``waveform``, computes
    output :math:`y` as

    .. math::
        y[i] = x[i] - \text{coeff} \cdot x[i - 1]

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Waveform, with shape `(..., N)`.
        coeff (float, optional): Pre-emphasis coefficient. Typically between 0.0 and 1.0.
            (Default: 0.97)

    Returns:
        torch.Tensor: Pre-emphasized waveform, with shape `(..., N)`.
    """
    waveform = waveform.clone()
    waveform[..., 1:] -= coeff * waveform[..., :-1]
    return waveform


def deemphasis(waveform, coeff: float = 0.97) -> torch.Tensor:
    r"""De-emphasizes a waveform along its last dimension.
    Inverse of :meth:`preemphasis`. Concretely, for each signal
    :math:`x` in ``waveform``, computes output :math:`y` as

    .. math::
        y[i] = x[i] + \text{coeff} \cdot y[i - 1]

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Waveform, with shape `(..., N)`.
        coeff (float, optional): De-emphasis coefficient. Typically between 0.0 and 1.0.
            (Default: 0.97)

    Returns:
        torch.Tensor: De-emphasized waveform, with shape `(..., N)`.
    """
    a_coeffs = torch.tensor([1.0, -coeff], dtype=waveform.dtype, device=waveform.device)
    b_coeffs = torch.tensor([1.0, 0.0], dtype=waveform.dtype, device=waveform.device)
    return torchaudio.functional.lfilter(waveform, a_coeffs=a_coeffs, b_coeffs=b_coeffs)


def frechet_distance(mu_x, sigma_x, mu_y, sigma_y):
    r"""Computes the Fréchet distance between two multivariate normal distributions :cite:`dowson1982frechet`.

    Concretely, for multivariate Gaussians :math:`X(\mu_X, \Sigma_X)`
    and :math:`Y(\mu_Y, \Sigma_Y)`, the function computes and returns :math:`F` as

    .. math::
        F(X, Y) = || \mu_X - \mu_Y ||_2^2
        + \text{Tr}\left( \Sigma_X + \Sigma_Y - 2 \sqrt{\Sigma_X \Sigma_Y} \right)

    Args:
        mu_x (torch.Tensor): mean :math:`\mu_X` of multivariate Gaussian :math:`X`, with shape `(N,)`.
        sigma_x (torch.Tensor): covariance matrix :math:`\Sigma_X` of :math:`X`, with shape `(N, N)`.
        mu_y (torch.Tensor): mean :math:`\mu_Y` of multivariate Gaussian :math:`Y`, with shape `(N,)`.
        sigma_y (torch.Tensor): covariance matrix :math:`\Sigma_Y` of :math:`Y`, with shape `(N, N)`.

    Returns:
        torch.Tensor: the Fréchet distance between :math:`X` and :math:`Y`.
    """
    if len(mu_x.size()) != 1:
        raise ValueError(f"Input mu_x must be one-dimensional; got dimension {len(mu_x.size())}.")
    if len(sigma_x.size()) != 2:
        raise ValueError(f"Input sigma_x must be two-dimensional; got dimension {len(sigma_x.size())}.")
    if sigma_x.size(0) != sigma_x.size(1) != mu_x.size(0):
        raise ValueError("Each of sigma_x's dimensions must match mu_x's size.")
    if mu_x.size() != mu_y.size():
        raise ValueError(f"Inputs mu_x and mu_y must have the same shape; got {mu_x.size()} and {mu_y.size()}.")
    if sigma_x.size() != sigma_y.size():
        raise ValueError(
            f"Inputs sigma_x and sigma_y must have the same shape; got {sigma_x.size()} and {sigma_y.size()}."
        )

    a = (mu_x - mu_y).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
    return a + b - 2 * c
