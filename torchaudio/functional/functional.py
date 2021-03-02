# -*- coding: utf-8 -*-

import io
import math
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torchaudio._internal import module_utils as _mod_utils
import torchaudio

__all__ = [
    "spectrogram",
    "griffinlim",
    "amplitude_to_DB",
    "DB_to_amplitude",
    "compute_deltas",
    "compute_kaldi_pitch",
    "create_fb_matrix",
    "create_dct",
    "compute_deltas",
    "detect_pitch_frequency",
    "DB_to_amplitude",
    "mu_law_encoding",
    "mu_law_decoding",
    "complex_norm",
    "angle",
    "magphase",
    "phase_vocoder",
    'mask_along_axis',
    'mask_along_axis_iid',
    'sliding_window_cmn',
    "spectral_centroid",
    "apply_codec",
]


def spectrogram(
        waveform: Tensor,
        pad: int,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: Optional[float],
        normalized: bool,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True
) -> Tensor:
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
    The spectrogram can be either magnitude-only or complex.

    Args:
        waveform (Tensor): Tensor of audio of dimension (..., time)
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (float or None): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
            If None, then the complex spectrum is returned instead.
        normalized (bool): Whether to normalize by magnitude after stft
        center (bool, optional): whether to pad :attr:`waveform` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy. Default: ``True``

    Returns:
        Tensor: Dimension (..., freq, time), freq is
        ``n_fft // 2 + 1`` and ``n_fft`` is the number of
        Fourier bins, and time is the number of window hops (n_frame).
    """

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

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
        normalized=False,
        onesided=onesided,
        return_complex=True,
    )

    # unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-2:])

    if normalized:
        spec_f /= window.pow(2.).sum().sqrt()
    if power is not None:
        if power == 1.0:
            return spec_f.abs()
        return spec_f.abs().pow(power)
    return torch.view_as_real(spec_f)


def griffinlim(
        specgram: Tensor,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: float,
        normalized: bool,
        n_iter: int,
        momentum: float,
        length: Optional[int],
        rand_init: bool
) -> Tensor:
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
        Implementation ported from `librosa`.

    *  [1] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto.
        "librosa: Audio and music signal analysis in python."
        In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
    *  [2] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.
    *  [3] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Args:
        specgram (Tensor): A magnitude-only STFT spectrogram of dimension (..., freq, frames)
            where freq is ``n_fft // 2 + 1``.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins
        hop_length (int): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        win_length (int): Window size. (Default: ``n_fft``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
        normalized (bool): Whether to normalize by magnitude after stft.
        n_iter (int): Number of iteration for phase recovery process.
        momentum (float): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge.
        length (int or None): Array length of the expected output.
        rand_init (bool): Initializes phase randomly if True, to zero otherwise.

    Returns:
        torch.Tensor: waveform of (..., time), where time equals the ``length`` parameter if given.
    """
    assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
    assert momentum >= 0, 'momentum={} < 0'.format(momentum)

    if normalized:
        warnings.warn(
            "The argument normalized is not used in Griffin-Lim, "
            "and will be removed in v0.9.0 release. To suppress this warning, "
            "please use `normalized=False`.")

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    specgram = specgram.pow(1 / power)

    # randomly initialize the phase
    batch, freq, frames = specgram.size()
    if rand_init:
        angles = 2 * math.pi * torch.rand(batch, freq, frames)
    else:
        angles = torch.zeros(batch, freq, frames)
    angles = torch.stack([angles.cos(), angles.sin()], dim=-1) \
        .to(dtype=specgram.dtype, device=specgram.device)
    specgram = specgram.unsqueeze(-1).expand_as(angles)

    # And initialize the previous iterate to 0
    rebuilt = torch.tensor(0.)

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = torch.istft(specgram * angles,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              win_length=win_length,
                              window=window,
                              length=length).float()

        # Rebuild the spectrogram
        rebuilt = torch.view_as_real(
            torch.stft(
                input=inverse,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                pad_mode='reflect',
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum / (1 + momentum))
        angles = angles.div(complex_norm(angles).add(1e-16).unsqueeze(-1).expand_as(angles))

    # Return the final phase estimates
    waveform = torch.istft(specgram * angles,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           win_length=win_length,
                           window=window,
                           length=length)

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform


def amplitude_to_DB(
        x: Tensor,
        multiplier: float,
        amin: float,
        db_multiplier: float,
        top_db: Optional[float] = None
) -> Tensor:
    r"""Turn a spectrogram from the power/amplitude scale to the decibel scale.

    The output of each tensor in a batch depends on the maximum value of that tensor,
    and so may return different values for an audio clip split into snippets vs. a full clip.

    Args:

        x (Tensor): Input spectrogram(s) before being converted to decibel scale. Input should take
          the form `(..., freq, time)`. Batched inputs should include a channel dimension and
          have the form `(batch, channel, freq, time)`.
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


def DB_to_amplitude(
        x: Tensor,
        ref: float,
        power: float
) -> Tensor:
    r"""Turn a tensor from the decibel scale to the power/amplitude scale.

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

    if mel_scale not in ['slaney', 'htk']:
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

    if mel_scale not in ['slaney', 'htk']:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = (mels >= min_log_mel)
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def create_fb_matrix(
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
) -> Tensor:
    r"""Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (Optional[str]): If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * create_fb_matrix(A.size(-1), ...)``.
    """

    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    if (fb.max(dim=0).values == 0.).any():
        warnings.warn(
            "At least one mel filterbank has all zero values. "
            f"The value for `n_mels` ({n_mels}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def create_dct(
        n_mfcc: int,
        n_mels: int,
        norm: Optional[str]
) -> Tensor:
    r"""Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.

    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either 'ortho' or None)

    Returns:
        Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == "ortho"
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def mu_law_encoding(
        x: Tensor,
        quantization_channels: int
) -> Tensor:
    r"""Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1.

    Args:
        x (Tensor): Input tensor
        quantization_channels (int): Number of channels

    Returns:
        Tensor: Input after mu-law encoding
    """
    mu = quantization_channels - 1.0
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


def mu_law_decoding(
        x_mu: Tensor,
        quantization_channels: int
) -> Tensor:
    r"""Decode mu-law encoded signal.  For more info see the
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


def complex_norm(
        complex_tensor: Tensor,
        power: float = 1.0
) -> Tensor:
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)


def angle(
        complex_tensor: Tensor
) -> Tensor:
    r"""Compute the angle of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`

    Return:
        Tensor: Angle of a complex tensor. Shape of `(..., )`
    """
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])


def magphase(
        complex_tensor: Tensor,
        power: float = 1.0
) -> Tuple[Tensor, Tensor]:
    r"""Separate a complex-valued spectrogram with shape `(..., 2)` into its magnitude and phase.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`)

    Returns:
        (Tensor, Tensor): The magnitude and phase of the complex tensor
    """
    mag = complex_norm(complex_tensor, power)
    phase = angle(complex_tensor)
    return mag, phase


def phase_vocoder(
        complex_specgrams: Tensor,
        rate: float,
        phase_advance: Tensor
) -> Tensor:
    r"""Given a STFT tensor, speed up in time without modifying pitch by a
    factor of ``rate``.

    Args:
        complex_specgrams (Tensor): Dimension of `(..., freq, time, complex=2)`
        rate (float): Speed-up factor
        phase_advance (Tensor): Expected phase advance in each bin. Dimension of (freq, 1)

    Returns:
        Tensor: Complex Specgrams Stretch with dimension of `(..., freq, ceil(time/rate), complex=2)`

    Example
        >>> freq, hop_length = 1025, 512
        >>> # (channel, freq, time, complex=2)
        >>> complex_specgrams = torch.randn(2, freq, 300, 2)
        >>> rate = 1.3 # Speed up by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, freq)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([2, 1025, 231, 2])
    """

    # pack batch
    shape = complex_specgrams.size()
    complex_specgrams = complex_specgrams.reshape([-1] + list(shape[-3:]))

    time_steps = torch.arange(0,
                              complex_specgrams.size(-2),
                              rate,
                              device=complex_specgrams.device,
                              dtype=complex_specgrams.dtype)

    alphas = time_steps % 1.0
    phase_0 = angle(complex_specgrams[..., :1, :])

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(complex_specgrams, [0, 0, 0, 2])

    # (new_bins, freq, 2)
    complex_specgrams_0 = complex_specgrams.index_select(-2, time_steps.long())
    complex_specgrams_1 = complex_specgrams.index_select(-2, (time_steps + 1).long())

    angle_0 = angle(complex_specgrams_0)
    angle_1 = angle(complex_specgrams_1)

    norm_0 = torch.norm(complex_specgrams_0, p=2, dim=-1)
    norm_1 = torch.norm(complex_specgrams_1, p=2, dim=-1)

    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[..., :-1]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    real_stretch = mag * torch.cos(phase_acc)
    imag_stretch = mag * torch.sin(phase_acc)

    complex_specgrams_stretch = torch.stack([real_stretch, imag_stretch], dim=-1)

    # unpack batch
    complex_specgrams_stretch = complex_specgrams_stretch.reshape(shape[:-3] + complex_specgrams_stretch.shape[1:])

    return complex_specgrams_stretch


def mask_along_axis_iid(
        specgrams: Tensor,
        mask_param: int,
        mask_value: float,
        axis: int
) -> Tensor:
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.

    Args:
        specgrams (Tensor): Real spectrograms (batch, channel, freq, time)
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (2 -> frequency, 3 -> time)

    Returns:
        Tensor: Masked spectrograms of dimensions (batch, channel, freq, time)
    """

    if axis != 2 and axis != 3:
        raise ValueError('Only Frequency and Time masking are supported')

    device = specgrams.device
    dtype = specgrams.dtype

    value = torch.rand(specgrams.shape[:2], device=device, dtype=dtype) * mask_param
    min_value = torch.rand(specgrams.shape[:2], device=device, dtype=dtype) * (specgrams.size(axis) - value)

    # Create broadcastable mask
    mask_start = min_value[..., None, None]
    mask_end = (min_value + value)[..., None, None]
    mask = torch.arange(0, specgrams.size(axis), device=device, dtype=dtype)

    # Per batch example masking
    specgrams = specgrams.transpose(axis, -1)
    specgrams.masked_fill_((mask >= mask_start) & (mask < mask_end), mask_value)
    specgrams = specgrams.transpose(axis, -1)

    return specgrams


def mask_along_axis(
        specgram: Tensor,
        mask_param: int,
        mask_value: float,
        axis: int
) -> Tensor:
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram (channel, freq, time)
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)

    Returns:
        Tensor: Masked spectrogram of dimensions (channel, freq, time)
    """

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()

    assert mask_end - mask_start < mask_param
    if axis == 1:
        specgram[:, mask_start:mask_end] = mask_value
    elif axis == 2:
        specgram[:, :, mask_start:mask_end] = mask_value
    else:
        raise ValueError('Only Frequency and Time masking are supported')

    # unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram


def compute_deltas(
        specgram: Tensor,
        win_length: int = 5,
        mode: str = "replicate"
) -> Tensor:
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. math::
       d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N}} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is ``(win_length-1)//2``.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: ``5``)
        mode (str, optional): Mode parameter passed to padding (Default: ``"replicate"``)

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)

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

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(specgram.shape[1], 1, 1)

    output = torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom

    # unpack batch
    output = output.reshape(shape)

    return output


def _compute_nccf(
        waveform: Tensor,
        sample_rate: int,
        frame_time: float,
        freq_low: int
) -> Tensor:
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
            / (EPSILON + torch.norm(s1, p=2, dim=-1)).pow(2)
            / (EPSILON + torch.norm(s2, p=2, dim=-1)).pow(2)
        )

        output_lag.append(output_frames.unsqueeze(-1))

    nccf = torch.cat(output_lag, -1)

    return nccf


def _combine_max(
        a: Tuple[Tensor, Tensor],
        b: Tuple[Tensor, Tensor],
        thresh: float = 0.99
) -> Tuple[Tensor, Tensor]:
    """
    Take value from first if bigger than a multiplicative factor of the second, elementwise.
    """
    mask = (a[0] > thresh * b[0])
    values = mask * a[0] + ~mask * b[0]
    indices = mask * a[1] + ~mask * b[1]
    return values, indices


def _find_max_per_frame(
        nccf: Tensor,
        sample_rate: int,
        freq_high: int
) -> Tensor:
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


def _median_smoothing(
        indices: Tensor,
        win_length: int
) -> Tensor:
    r"""
    Apply median smoothing to the 1D tensor over the given window.
    """

    # Centered windowed
    pad_length = (win_length - 1) // 2

    # "replicate" padding in any dimension
    indices = torch.nn.functional.pad(
        indices, (pad_length, 0), mode="constant", value=0.
    )

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

    It is implemented using normalized cross-correlation function and median smoothing.

    Args:
        waveform (Tensor): Tensor of audio of dimension (..., freq, time)
        sample_rate (int): The sample rate of the waveform (Hz)
        frame_time (float, optional): Duration of a frame (Default: ``10 ** (-2)``).
        win_length (int, optional): The window length for median smoothing (in number of frames) (Default: ``30``).
        freq_low (int, optional): Lowest frequency that can be detected (Hz) (Default: ``85``).
        freq_high (int, optional): Highest frequency that can be detected (Hz) (Default: ``3400``).

    Returns:
        Tensor: Tensor of freq of dimension (..., frame)
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
    waveform: Tensor,
    cmn_window: int = 600,
    min_cmn_window: int = 100,
    center: bool = False,
    norm_vars: bool = False,
) -> Tensor:
    r"""
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    Args:
        waveform (Tensor): Tensor of audio of dimension (..., freq, time)
        cmn_window (int, optional): Window in frames for running average CMN computation (int, default = 600)
        min_cmn_window (int, optional):  Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center == false, ignored if center==true (int, default = 100)
        center (bool, optional): If true, use a window centered on the current frame
            (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
        norm_vars (bool, optional): If true, normalize variance to one. (bool, default = false)

    Returns:
        Tensor: Tensor of freq of dimension (..., frame)
    """
    input_shape = waveform.shape
    num_frames, num_feats = input_shape[-2:]
    waveform = waveform.view(-1, num_frames, num_feats)
    num_channels = waveform.shape[0]

    dtype = waveform.dtype
    device = waveform.device
    last_window_start = last_window_end = -1
    cur_sum = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
    cur_sumsq = torch.zeros(num_channels, num_feats, dtype=dtype, device=device)
    cmn_waveform = torch.zeros(
        num_channels, num_frames, num_feats, dtype=dtype, device=device)
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
            window_start -= (window_end - num_frames)
            window_end = num_frames
            if window_start < 0:
                window_start = 0
        if last_window_start == -1:
            input_part = waveform[:, window_start: window_end - window_start, :]
            cur_sum += torch.sum(input_part, 1)
            if norm_vars:
                cur_sumsq += torch.cumsum(input_part ** 2, 1)[:, -1, :]
        else:
            if window_start > last_window_start:
                frame_to_remove = waveform[:, last_window_start, :]
                cur_sum -= frame_to_remove
                if norm_vars:
                    cur_sumsq -= (frame_to_remove ** 2)
            if window_end > last_window_end:
                frame_to_add = waveform[:, last_window_end, :]
                cur_sum += frame_to_add
                if norm_vars:
                    cur_sumsq += (frame_to_add ** 2)
        window_frames = window_end - window_start
        last_window_start = window_start
        last_window_end = window_end
        cmn_waveform[:, t, :] = waveform[:, t, :] - cur_sum / window_frames
        if norm_vars:
            if window_frames == 1:
                cmn_waveform[:, t, :] = torch.zeros(
                    num_channels, num_feats, dtype=dtype, device=device)
            else:
                variance = cur_sumsq
                variance = variance / window_frames
                variance -= ((cur_sum ** 2) / (window_frames ** 2))
                variance = torch.pow(variance, -0.5)
                cmn_waveform[:, t, :] *= variance

    cmn_waveform = cmn_waveform.view(input_shape[:-2] + (num_frames, num_feats))
    if len(input_shape) == 2:
        cmn_waveform = cmn_waveform.squeeze(0)
    return cmn_waveform


def spectral_centroid(
        waveform: Tensor,
        sample_rate: int,
        pad: int,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
) -> Tensor:
    r"""
    Compute the spectral centroid for each channel along the time axis.

    The spectral centroid is defined as the weighted average of the
    frequency values, weighted by their magnitude.

    Args:
        waveform (Tensor): Tensor of audio of dimension (..., time)
        sample_rate (int): Sample rate of the audio waveform
        pad (int): Two sided padding of signal
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size

    Returns:
        Tensor: Dimension (..., time)
    """
    specgram = spectrogram(waveform, pad=pad, window=window, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, power=1., normalized=False)
    freqs = torch.linspace(0, sample_rate // 2, steps=1 + n_fft // 2,
                           device=specgram.device).reshape((-1, 1))
    freq_dim = -2
    return (freqs * specgram).sum(dim=freq_dim) / specgram.sum(dim=freq_dim)


@_mod_utils.requires_sox()
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

    Args:
        waveform (Tensor): Audio data. Must be 2 dimensional. See also ```channels_first```.
        sample_rate (int): Sample rate of the audio waveform.
        format (str): File format.
        channels_first (bool):
            When True, both the input and output Tensor have dimension ``[channel, time]``.
            Otherwise, they have dimension ``[time, channel]``.
        compression (float): Used for formats other than WAV.
            For mor details see :py:func:`torchaudio.backend.sox_io_backend.save`.
        encoding (str, optional): Changes the encoding for the supported formats.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.
        bits_per_sample (int, optional): Changes the bit depth for the supported formats.
            For more details see :py:func:`torchaudio.backend.sox_io_backend.save`.

    Returns:
        torch.Tensor: Resulting Tensor.
        If ``channels_first=True``, it has ``[channel, time]`` else ``[time, channel]``.
    """
    bytes = io.BytesIO()
    torchaudio.backend.sox_io_backend.save(bytes,
                                           waveform,
                                           sample_rate,
                                           channels_first,
                                           compression,
                                           format,
                                           encoding,
                                           bits_per_sample
                                           )
    bytes.seek(0)
    augmented, _ = torchaudio.sox_effects.sox_effects.apply_effects_file(
        bytes, effects=[["rate", f"{sample_rate}"]], channels_first=channels_first, format=format)
    return augmented


def compute_kaldi_pitch(
        waveform: torch.Tensor,
        sample_rate: float,
        frame_length: float = 25.0,
        frame_shift: float = 10.0,
        min_f0: float = 50,
        max_f0: float = 400,
        soft_min_f0: float = 10.0,
        penalty_factor: float = 0.1,
        lowpass_cutoff: float = 1000,
        resample_frequency: float = 4000,
        delta_pitch: float = 0.005,
        nccf_ballast: float = 7000,
        lowpass_filter_width: int = 1,
        upsample_filter_width: int = 5,
        max_frames_latency: int = 0,
        frames_per_chunk: int = 0,
        simulate_first_pass_online: bool = False,
        recompute_frame: int = 500,
        snip_edges: bool = True,
) -> torch.Tensor:
    """Extract pitch based on method described in [1].

    This function computes the equivalent of `compute-kaldi-pitch-feats` from Kaldi.

    Args:
        waveform (Tensor):
            The input waveform of shape `(..., time)`.
        sample_rate (float):
            Sample rate of `waveform`.
        frame_length (float, optional):
            Frame length in milliseconds. (default: 25.0)
        frame_shift (float, optional):
            Frame shift in milliseconds. (default: 10.0)
        min_f0 (float, optional):
            Minimum F0 to search for (Hz)  (default: 50.0)
        max_f0 (float, optional):
            Maximum F0 to search for (Hz)  (default: 400.0)
        soft_min_f0 (float, optional):
            Minimum f0, applied in soft way, must not exceed min-f0  (default: 10.0)
        penalty_factor (float, optional):
            Cost factor for FO change.  (default: 0.1)
        lowpass_cutoff (float, optional):
            Cutoff frequency for LowPass filter (Hz) (default: 1000)
        resample_frequency (float, optional):
            Frequency that we down-sample the signal to. Must be more than twice lowpass-cutoff.
            (default: 4000)
        delta_pitch( float, optional):
            Smallest relative change in pitch that our algorithm measures. (default: 0.005)
        nccf_ballast (float, optional):
            Increasing this factor reduces NCCF for quiet frames (default: 7000)
        lowpass_filter_width (int, optional):
            Integer that determines filter width of lowpass filter, more gives sharper filter.
            (default: 1)
        upsample_filter_width (int, optional):
            Integer that determines filter width when upsampling NCCF. (default: 5)
        max_frames_latency (int, optional):
            Maximum number of frames of latency that we allow pitch tracking to introduce into
            the feature processing (affects output only if ``frames_per_chunk > 0`` and
            ``simulate_first_pass_online=True``) (default: 0)
        frames_per_chunk (int, optional):
            The number of frames used for energy normalization. (default: 0)
        simulate_first_pass_online (bool, optional):
            If true, the function will output features that correspond to what an online decoder
            would see in the first pass of decoding -- not the final version of the features,
            which is the default. (default: False)
            Relevant if ``frames_per_chunk > 0``.
        recompute_frame (int, optional):
            Only relevant for compatibility with online pitch extraction.
            A non-critical parameter; the frame at which we recompute some of the forward pointers,
            after revising our estimate of the signal energy.
            Relevant if ``frames_per_chunk > 0``. (default: 500)
        snip_edges (bool, optional):
            If this is set to false, the incomplete frames near the ending edge won't be snipped,
            so that the number of frames is the file size divided by the frame-shift.
            This makes different types of features give the same number of frames. (default: True)

    Returns:
       Tensor: Pitch feature. Shape: ``(batch, frames 2)`` where the last dimension
       corresponds to pitch and NCCF.

    Reference:
        - A pitch extraction algorithm tuned for automatic speech recognition

          P. Ghahremani, B. BabaAli, D. Povey, K. Riedhammer, J. Trmal and S. Khudanpur

          2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),

          Florence, 2014, pp. 2494-2498, doi: 10.1109/ICASSP.2014.6854049.
    """
    shape = waveform.shape
    waveform = waveform.reshape(-1, shape[-1])
    result = torch.ops.torchaudio.kaldi_ComputeKaldiPitch(
        waveform, sample_rate, frame_length, frame_shift,
        min_f0, max_f0, soft_min_f0, penalty_factor, lowpass_cutoff,
        resample_frequency, delta_pitch, nccf_ballast,
        lowpass_filter_width, upsample_filter_width, max_frames_latency,
        frames_per_chunk, simulate_first_pass_online, recompute_frame,
        snip_edges,
    )
    result = result.reshape(shape[:-1] + result.shape[-2:])
    return result
