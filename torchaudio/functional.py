# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple
import warnings

import torch
from torch import Tensor

__all__ = [
    "istft",
    "spectrogram",
    "griffinlim",
    "amplitude_to_DB",
    "create_fb_matrix",
    "create_dct",
    "mu_law_encoding",
    "mu_law_decoding",
    "complex_norm",
    "angle",
    "magphase",
    "phase_vocoder",
    "lfilter",
    "lowpass_biquad",
    "highpass_biquad",
    "allpass_biquad",
    "bandpass_biquad",
    "bandreject_biquad",
    "equalizer_biquad",
    "band_biquad",
    "treble_biquad",
    "bass_biquad",
    "deemph_biquad",
    "riaa_biquad",
    "biquad",
    "contrast",
    "dcshift",
    "overdrive",
    "phaser",
    "flanger",
    'mask_along_axis',
    'mask_along_axis_iid',
    'sliding_window_cmn',
    'vad',
]


def istft(
        stft_matrix: Tensor,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[Tensor] = None,
        center: bool = True,
        pad_mode: Optional[str] = None,
        normalized: bool = False,
        onesided: bool = True,
        length: Optional[int] = None,
) -> Tensor:
    r"""Inverse short time Fourier Transform. This is expected to be the inverse of torch.stft.
    It has the same parameters (+ additional optional parameter of ``length``) and it should return the
    least squares estimation of the original signal. The algorithm will check using the NOLA condition (
    nonzero overlap).

    Important consideration in the parameters ``window`` and ``center`` so that the envelop
    created by the summation of all the windows is never zero at certain point in time. Specifically,
    :math:`\sum_{t=-\infty}^{\infty} w^2[n-t\times hop\_length] \cancel{=} 0`.

    Since stft discards elements at the end of the signal if they do not fit in a frame, the
    istft may return a shorter signal than the original signal (can occur if ``center`` is False
    since the signal isn't padded).

    If ``center`` is True, then there will be padding e.g. 'constant', 'reflect', etc. Left padding
    can be trimmed off exactly because they can be calculated but right padding cannot be calculated
    without additional information.

    Example: Suppose the last window is:
    [17, 18, 0, 0, 0] vs [18, 0, 0, 0, 0]

    The n_frame, hop_length, win_length are all the same which prevents the calculation of right padding.
    These additional values could be zeros or a reflection of the signal so providing ``length``
    could be useful. If ``length`` is ``None`` then padding will be aggressively removed
    (some loss of signal).

    [1] D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier transform,"
    IEEE Trans. ASSP, vol.32, no.2, pp.236-243, Apr. 1984.

    Args:
        stft_matrix (Tensor): Output of stft where each row of a channel is a frequency and each
            column is a window. It has a size of either (..., fft_size, n_frame, 2)
        n_fft (int): Size of Fourier transform
        hop_length (int or None, optional): The distance between neighboring sliding window frames.
            (Default: ``win_length // 4``)
        win_length (int or None, optional): The size of window frame and STFT filter. (Default: ``n_fft``)
        window (Tensor or None, optional): The optional window function.
            (Default: ``torch.ones(win_length)``)
        center (bool, optional): Whether ``input`` was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            (Default: ``True``)
        pad_mode: This argument was ignored and to be removed.
        normalized (bool, optional): Whether the STFT was normalized. (Default: ``False``)
        onesided (bool, optional): Whether the STFT is onesided. (Default: ``True``)
        length (int or None, optional): The amount to trim the signal by (i.e. the
            original signal length). (Default: whole signal)

    Returns:
        Tensor: Least squares estimation of the original signal of size (..., signal_length)
    """
    warnings.warn(
        'istft has been moved to PyTorch and will be removed from torchaudio, '
        'please use torch.istft instead.')
    if pad_mode is not None:
        warnings.warn(
            'The parameter `pad_mode` was ignored in isftft, and is thus being deprecated. '
            'Please set `pad_mode` to None to suppress this warning.')
    return torch.istft(
        input=stft_matrix, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
        center=center, normalized=normalized, onesided=onesided, length=length)


def spectrogram(
        waveform: Tensor,
        pad: int,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: Optional[float],
        normalized: bool
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
        waveform, n_fft, hop_length, win_length, window, True, "reflect", False, True
    )

    # unpack batch
    spec_f = spec_f.reshape(shape[:-1] + spec_f.shape[-3:])

    if normalized:
        spec_f /= window.pow(2.).sum().sqrt()
    if power is not None:
        spec_f = complex_norm(spec_f, power=power)

    return spec_f


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
        inverse = istft(specgram * angles,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window,
                        length=length).float()

        # Rebuild the spectrogram
        rebuilt = torch.stft(inverse, n_fft, hop_length, win_length, window,
                             True, 'reflect', False, True)

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum / (1 + momentum))
        angles = angles.div(complex_norm(angles).add(1e-16).unsqueeze(-1).expand_as(angles))

    # Return the final phase estimates
    waveform = istft(specgram * angles,
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
    r"""Turn a tensor from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input tensor, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        x (Tensor): Input tensor before being converted to decibel scale
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
        x_db = x_db.clamp(min=x_db.max().item() - top_db)

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


def create_fb_matrix(
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        norm: Optional[str] = None
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
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
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


def lfilter(
        waveform: Tensor,
        a_coeffs: Tensor,
        b_coeffs: Tensor,
        clamp: bool = True,
) -> Tensor:
    r"""Perform an IIR filter by evaluating difference equation.

    Args:
        waveform (Tensor): audio waveform of dimension of ``(..., time)``.  Must be normalized to -1 to 1.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of ``(n_order + 1)``.
                                Lower delays coefficients are first, e.g. ``[a0, a1, a2, ...]``.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of ``(n_order + 1)``.
                                 Lower delays coefficients are first, e.g. ``[b0, b1, b2, ...]``.
                                 Must be same size as a_coeffs (pad with 0's as necessary).
        clamp (bool, optional): If ``True``, clamp the output signal to be in the range [-1, 1] (Default: ``True``)

    Returns:
        Tensor: Waveform with dimension of ``(..., time)``.
    """
    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    assert (a_coeffs.size(0) == b_coeffs.size(0))
    assert (len(waveform.size()) == 2)
    assert (waveform.device == a_coeffs.device)
    assert (b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(0)
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(0)
    b_coeffs_flipped = b_coeffs.flip(0)

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)
    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()
    # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
    input_signal_windows = torch.matmul(b_coeffs_flipped, torch.take(padded_waveform, window_idxs))

    input_signal_windows.div_(a_coeffs[0])
    a_coeffs_flipped.div_(a_coeffs[0])
    for i_sample, o0 in enumerate(input_signal_windows.t()):
        windowed_output_signal = padded_output_waveform[:, i_sample:(i_sample + n_order)]
        o0.addmv_(windowed_output_signal, a_coeffs_flipped, alpha=-1)
        padded_output_waveform[:, i_sample + n_order - 1] = o0

    output = padded_output_waveform[:, (n_order - 1):]

    if clamp:
        output = torch.clamp(output, min=-1., max=1.)

    # unpack batch
    output = output.reshape(shape[:-1] + output.shape[-1:])

    return output


def biquad(
        waveform: Tensor,
        b0: float,
        b1: float,
        b2: float,
        a0: float,
        a1: float,
        a2: float
) -> Tensor:
    r"""Perform a biquad filter of input tensor.  Initial conditions set to 0.
    https://en.wikipedia.org/wiki/Digital_biquad_filter

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        b0 (float): numerator coefficient of current input, x[n]
        b1 (float): numerator coefficient of input one time step ago x[n-1]
        b2 (float): numerator coefficient of input two time steps ago x[n-2]
        a0 (float): denominator coefficient of current output y[n], typically 1
        a1 (float): denominator coefficient of current output y[n-1]
        a2 (float): denominator coefficient of current output y[n-2]

    Returns:
        Tensor: Waveform with dimension of `(..., time)`
    """

    device = waveform.device
    dtype = waveform.dtype

    output_waveform = lfilter(
        waveform,
        torch.tensor([a0, a1, a2], dtype=dtype, device=device),
        torch.tensor([b0, b1, b2], dtype=dtype, device=device)
    )
    return output_waveform


def _dB2Linear(x: float) -> float:
    return math.exp(x * math.log(10) / 20.0)


def highpass_biquad(
        waveform: Tensor,
        sample_rate: int,
        cutoff_freq: float,
        Q: float = 0.707
) -> Tensor:
    r"""Design biquad highpass filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform dimension of `(..., time)`
    """
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2. / Q

    b0 = (1 + math.cos(w0)) / 2
    b1 = -1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def lowpass_biquad(
        waveform: Tensor,
        sample_rate: int,
        cutoff_freq: float,
        Q: float = 0.707
) -> Tensor:
    r"""Design biquad lowpass filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (torch.Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        cutoff_freq (float): filter cutoff frequency
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`
    """
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def allpass_biquad(
        waveform: Tensor,
        sample_rate: int,
        central_freq: float,
        Q: float = 0.707
) -> Tensor:
    r"""Design two-pole all-pass filter.  Similar to SoX implementation.

    Args:
        waveform(torch.Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        central_freq (float): central frequency (in Hz)
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q

    b0 = 1 - alpha
    b1 = -2 * math.cos(w0)
    b2 = 1 + alpha
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def bandpass_biquad(
        waveform: Tensor,
        sample_rate: int,
        central_freq: float,
        Q: float = 0.707,
        const_skirt_gain: bool = False
) -> Tensor:
    r"""Design two-pole band-pass filter.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        central_freq (float): central frequency (in Hz)
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)
        const_skirt_gain (bool, optional) : If ``True``, uses a constant skirt gain (peak gain = Q).
            If ``False``, uses a constant 0dB peak gain. (Default: ``False``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q

    temp = math.sin(w0) / 2 if const_skirt_gain else alpha
    b0 = temp
    b1 = 0.
    b2 = -temp
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def bandreject_biquad(
        waveform: Tensor,
        sample_rate: int,
        central_freq: float,
        Q: float = 0.707
) -> Tensor:
    r"""Design two-pole band-reject filter.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        central_freq (float): central frequency (in Hz)
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q

    b0 = 1.
    b1 = -2 * math.cos(w0)
    b2 = 1.
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def equalizer_biquad(
        waveform: Tensor,
        sample_rate: int,
        center_freq: float,
        gain: float,
        Q: float = 0.707
) -> Tensor:
    r"""Design biquad peaking equalizer filter and perform filtering.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        center_freq (float): filter's central frequency
        gain (float): desired gain at the boost (or attenuation) in dB
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`
    """
    w0 = 2 * math.pi * center_freq / sample_rate
    A = math.exp(gain / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q

    b0 = 1 + alpha * A
    b1 = -2 * math.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha / A
    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def band_biquad(
        waveform: Tensor,
        sample_rate: int,
        central_freq: float,
        Q: float = 0.707,
        noise: bool = False
) -> Tensor:
    r"""Design two-pole band filter.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        central_freq (float): central frequency (in Hz)
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``).
        noise (bool, optional) : If ``True``, uses the alternate mode for un-pitched audio (e.g. percussion).
            If ``False``, uses mode oriented to pitched audio, i.e. voice, singing,
            or instrumental music (Default: ``False``).

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    w0 = 2 * math.pi * central_freq / sample_rate
    bw_Hz = central_freq / Q

    a0 = 1.
    a2 = math.exp(-2 * math.pi * bw_Hz / sample_rate)
    a1 = -4 * a2 / (1 + a2) * math.cos(w0)

    b0 = math.sqrt(1 - a1 * a1 / (4 * a2)) * (1 - a2)

    if noise:
        mult = math.sqrt(((1 + a2) * (1 + a2) - a1 * a1) * (1 - a2) / (1 + a2)) / b0
        b0 *= mult

    b1 = 0.
    b2 = 0.

    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def treble_biquad(
        waveform: Tensor,
        sample_rate: int,
        gain: float,
        central_freq: float = 3000,
        Q: float = 0.707
) -> Tensor:
    r"""Design a treble tone-control effect.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        gain (float): desired gain at the boost (or attenuation) in dB.
        central_freq (float, optional): central frequency (in Hz). (Default: ``3000``)
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``).

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    A = math.exp(gain / 40 * math.log(10))

    temp1 = 2 * math.sqrt(A) * alpha
    temp2 = (A - 1) * math.cos(w0)
    temp3 = (A + 1) * math.cos(w0)

    b0 = A * ((A + 1) + temp2 + temp1)
    b1 = -2 * A * ((A - 1) + temp3)
    b2 = A * ((A + 1) + temp2 - temp1)
    a0 = (A + 1) - temp2 + temp1
    a1 = 2 * ((A - 1) - temp3)
    a2 = (A + 1) - temp2 - temp1

    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def bass_biquad(
        waveform: Tensor,
        sample_rate: int,
        gain: float,
        central_freq: float = 100,
        Q: float = 0.707
) -> Tensor:
    r"""Design a bass tone-control effect.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        gain (float): desired gain at the boost (or attenuation) in dB.
        central_freq (float, optional): central frequency (in Hz). (Default: ``100``)
        Q (float, optional): https://en.wikipedia.org/wiki/Q_factor (Default: ``0.707``).

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """
    w0 = 2 * math.pi * central_freq / sample_rate
    alpha = math.sin(w0) / 2 / Q
    A = math.exp(gain / 40 * math.log(10))

    temp1 = 2 * math.sqrt(A) * alpha
    temp2 = (A - 1) * math.cos(w0)
    temp3 = (A + 1) * math.cos(w0)

    b0 = A * ((A + 1) - temp2 + temp1)
    b1 = 2 * A * ((A - 1) - temp3)
    b2 = A * ((A + 1) - temp2 - temp1)
    a0 = (A + 1) + temp2 + temp1
    a1 = -2 * ((A - 1) + temp3)
    a2 = (A + 1) + temp2 - temp1

    return biquad(waveform, b0 / a0, b1 / a0, b2 / a0, a0 / a0, a1 / a0, a2 / a0)


def deemph_biquad(
        waveform: Tensor,
        sample_rate: int
) -> Tensor:
    r"""Apply ISO 908 CD de-emphasis (shelving) IIR filter.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, Allowed sample rate ``44100`` or ``48000``

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """

    if sample_rate == 44100:
        central_freq = 5283
        width_slope = 0.4845
        gain = -9.477
    elif sample_rate == 48000:
        central_freq = 5356
        width_slope = 0.479
        gain = -9.62
    else:
        raise ValueError("Sample rate must be 44100 (audio-CD) or 48000 (DAT)")

    w0 = 2 * math.pi * central_freq / sample_rate
    A = math.exp(gain / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 * math.sqrt((A + 1 / A) * (1 / width_slope - 1) + 2)

    temp1 = 2 * math.sqrt(A) * alpha
    temp2 = (A - 1) * math.cos(w0)
    temp3 = (A + 1) * math.cos(w0)

    b0 = A * ((A + 1) + temp2 + temp1)
    b1 = -2 * A * ((A - 1) + temp3)
    b2 = A * ((A + 1) + temp2 - temp1)
    a0 = (A + 1) - temp2 + temp1
    a1 = 2 * ((A - 1) - temp3)
    a2 = (A + 1) - temp2 - temp1

    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def riaa_biquad(
        waveform: Tensor,
        sample_rate: int
) -> Tensor:
    r"""Apply RIAA vinyl playback equalisation.  Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz).
            Allowed sample rates in Hz : ``44100``,``48000``,``88200``,``96000``

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        https://www.w3.org/2011/audio/audio-eq-cookbook.html#APF
    """

    if (sample_rate == 44100):
        zeros = [-0.2014898, 0.9233820]
        poles = [0.7083149, 0.9924091]

    elif (sample_rate == 48000):
        zeros = [-0.1766069, 0.9321590]
        poles = [0.7396325, 0.9931330]

    elif (sample_rate == 88200):
        zeros = [-0.1168735, 0.9648312]
        poles = [0.8590646, 0.9964002]

    elif (sample_rate == 96000):
        zeros = [-0.1141486, 0.9676817]
        poles = [0.8699137, 0.9966946]

    else:
        raise ValueError("Sample rate must be 44.1k, 48k, 88.2k, or 96k")

    # polynomial coefficients with roots zeros[0] and zeros[1]
    b0 = 1.
    b1 = -(zeros[0] + zeros[1])
    b2 = (zeros[0] * zeros[1])

    # polynomial coefficients with roots poles[0] and poles[1]
    a0 = 1.
    a1 = -(poles[0] + poles[1])
    a2 = (poles[0] * poles[1])

    # Normalise to 0dB at 1kHz
    y = 2 * math.pi * 1000 / sample_rate
    b_re = b0 + b1 * math.cos(-y) + b2 * math.cos(-2 * y)
    a_re = a0 + a1 * math.cos(-y) + a2 * math.cos(-2 * y)
    b_im = b1 * math.sin(-y) + b2 * math.sin(-2 * y)
    a_im = a1 * math.sin(-y) + a2 * math.sin(-2 * y)
    g = 1 / math.sqrt((b_re ** 2 + b_im ** 2) / (a_re ** 2 + a_im ** 2))

    b0 *= g
    b1 *= g
    b2 *= g

    return biquad(waveform, b0, b1, b2, a0, a1, a2)


def contrast(
        waveform: Tensor,
        enhancement_amount: float = 75.
) -> Tensor:
    r"""Apply contrast effect.  Similar to SoX implementation.
    Comparable with compression, this effect modifies an audio signal to make it sound louder

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        enhancement_amount (float): controls the amount of the enhancement
            Allowed range of values for enhancement_amount : 0-100
            Note that enhancement_amount = 0 still gives a significant contrast enhancement

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
    """

    if not 0 <= enhancement_amount <= 100:
        raise ValueError("Allowed range of values for enhancement_amount : 0-100")

    contrast = enhancement_amount / 750.

    temp1 = waveform * (math.pi / 2)
    temp2 = contrast * torch.sin(temp1 * 4)
    output_waveform = torch.sin(temp1 + temp2)

    return output_waveform


def dcshift(
        waveform: Tensor,
        shift: float,
        limiter_gain: Optional[float] = None
) -> Tensor:
    r"""Apply a DC shift to the audio. Similar to SoX implementation.
    This can be useful to remove a DC offset
    (caused perhaps by a hardware problem in the recording chain) from the audio

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        shift (float): indicates the amount to shift the audio
            Allowed range of values for shift : -2.0 to +2.0
        limiter_gain (float): It is used only on peaks to prevent clipping
            It should have a value much less than 1 (e.g. 0.05 or 0.02)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
    """
    output_waveform = waveform
    limiter_threshold = 0.

    if limiter_gain is not None:
        limiter_threshold = 1.0 - (abs(shift) - limiter_gain)

    if limiter_gain is not None and shift > 0:
        mask = waveform > limiter_threshold
        temp = (waveform[mask] - limiter_threshold) * limiter_gain / (1 - limiter_threshold)
        output_waveform[mask] = (temp + limiter_threshold + shift).clamp(max=limiter_threshold)
        output_waveform[~mask] = (waveform[~mask] + shift).clamp(min=-1, max=1)
    elif limiter_gain is not None and shift < 0:
        mask = waveform < -limiter_threshold
        temp = (waveform[mask] + limiter_threshold) * limiter_gain / (1 - limiter_threshold)
        output_waveform[mask] = (temp - limiter_threshold + shift).clamp(min=-limiter_threshold)
        output_waveform[~mask] = (waveform[~mask] + shift).clamp(min=-1, max=1)
    else:
        output_waveform = (waveform + shift).clamp(min=-1, max=1)

    return output_waveform


def overdrive(
        waveform: Tensor,
        gain: float = 20,
        colour: float = 20
) -> Tensor:
    r"""Apply a overdrive effect to the audio. Similar to SoX implementation.
    This effect applies a non linear distortion to the audio signal.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        gain (float): desired gain at the boost (or attenuation) in dB
            Allowed range of values are 0 to 100
        colour (float):  controls the amount of even harmonic content in the over-driven output
            Allowed range of values are 0 to 100

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
    """
    actual_shape = waveform.shape
    device, dtype = waveform.device, waveform.dtype

    # convert to 2D (..,time)
    waveform = waveform.view(-1, actual_shape[-1])

    gain = _dB2Linear(gain)
    colour = colour / 200
    last_in = torch.zeros(waveform.shape[:-1], dtype=dtype, device=device)
    last_out = torch.zeros(waveform.shape[:-1], dtype=dtype, device=device)

    temp = waveform * gain + colour

    mask1 = temp < -1
    temp[mask1] = torch.tensor(-2.0 / 3.0, dtype=dtype, device=device)
    # Wrapping the constant with Tensor is required for Torchscript

    mask2 = temp > 1
    temp[mask2] = torch.tensor(2.0 / 3.0, dtype=dtype, device=device)

    mask3 = (~mask1 & ~mask2)
    temp[mask3] = temp[mask3] - (temp[mask3]**3) * (1. / 3)

    output_waveform = torch.zeros_like(waveform, dtype=dtype, device=device)

    # TODO: Implement a torch CPP extension
    for i in range(waveform.shape[-1]):
        last_out = temp[:, i] - last_in + 0.995 * last_out
        last_in = temp[:, i]
        output_waveform[:, i] = waveform[:, i] * 0.5 + last_out * 0.75

    return output_waveform.clamp(min=-1, max=1).view(actual_shape)


def phaser(
        waveform: Tensor,
        sample_rate: int,
        gain_in: float = 0.4,
        gain_out: float = 0.74,
        delay_ms: float = 3.0,
        decay: float = 0.4,
        mod_speed: float = 0.5,
        sinusoidal: bool = True
) -> Tensor:
    r"""Apply a phasing effect to the audio. Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        gain_in (float): desired input gain at the boost (or attenuation) in dB
            Allowed range of values are 0 to 1
        gain_out (float): desired output gain at the boost (or attenuation) in dB
            Allowed range of values are 0 to 1e9
        delay_ms (float): desired delay in milli seconds
            Allowed range of values are 0 to 5.0
        decay (float):  desired decay relative to gain-in
            Allowed range of values are 0 to 0.99
        mod_speed (float):  modulation speed in Hz
            Allowed range of values are 0.1 to 2
        sinusoidal (bool):  If ``True``, uses sinusoidal modulation (preferable for multiple instruments)
            If ``False``, uses triangular modulation (gives single instruments a sharper phasing effect)
            (Default: ``True``)

    Returns:
        Tensor: Waveform of dimension of `(..., time)`

    References:
        http://sox.sourceforge.net/sox.html
        Scott Lehman, Effects Explained, http://harmony-central.com/Effects/effects-explained.html
    """
    actual_shape = waveform.shape
    device, dtype = waveform.device, waveform.dtype

    # convert to 2D (channels,time)
    waveform = waveform.view(-1, actual_shape[-1])

    delay_buf_len = int((delay_ms * .001 * sample_rate) + .5)
    delay_buf = torch.zeros(waveform.shape[0], delay_buf_len, dtype=dtype, device=device)

    mod_buf_len = int(sample_rate / mod_speed + .5)

    if sinusoidal:
        wave_type = 'SINE'
    else:
        wave_type = 'TRIANGLE'

    mod_buf = _generate_wave_table(wave_type=wave_type,
                                   data_type='INT',
                                   table_size=mod_buf_len,
                                   min=1.,
                                   max=float(delay_buf_len),
                                   phase=math.pi / 2,
                                   device=device)

    delay_pos = 0
    mod_pos = 0

    output_waveform = torch.zeros_like(waveform, dtype=dtype, device=device)

    for i in range(waveform.shape[-1]):
        idx = int((delay_pos + mod_buf[mod_pos]) % delay_buf_len)
        temp = (waveform[:, i] * gain_in) + (delay_buf[:, idx] * decay)
        mod_pos = (mod_pos + 1) % mod_buf_len
        delay_pos = (delay_pos + 1) % delay_buf_len
        delay_buf[:, delay_pos] = temp
        output_waveform[:, i] = temp * gain_out

    return output_waveform.clamp(min=-1, max=1).view(actual_shape)


def _generate_wave_table(
        wave_type: str,
        data_type: str,
        table_size: int,
        min: float,
        max: float,
        phase: float,
        device: torch.device
) -> Tensor:
    r"""A helper fucntion for phaser. Generates a table with given parameters

    Args:
        wave_type (str): SINE or TRIANGULAR
        data_type (str): desired data_type ( `INT` or `FLOAT` )
        table_size (int): desired table size
        min (float): desired min value
        max (float): desired max value
        phase (float): desired phase
        device (torch.device): Torch device on which table must be generated
    Returns:
        Tensor: A 1D tensor with wave table values
    """

    phase_offset = int(phase / math.pi / 2 * table_size + 0.5)

    t = torch.arange(table_size, device=device, dtype=torch.int32)

    point = (t + phase_offset) % table_size

    d = torch.zeros_like(point, device=device, dtype=torch.float64)

    if wave_type == 'SINE':
        d = (torch.sin(point.to(torch.float64) / table_size * 2 * math.pi) + 1) / 2
    elif wave_type == 'TRIANGLE':
        d = point.to(torch.float64) * 2 / table_size
        value = 4 * point // table_size
        d[value == 0] = d[value == 0] + 0.5
        d[value == 1] = 1.5 - d[value == 1]
        d[value == 2] = 1.5 - d[value == 2]
        d[value == 3] = d[value == 3] - 1.5

    d = d * (max - min) + min

    if data_type == 'INT':
        mask = d < 0
        d[mask] = d[mask] - 0.5
        d[~mask] = d[~mask] + 0.5
        d = d.to(torch.int32)
    elif data_type == 'FLOAT':
        d = d.to(torch.float32)

    return d


def flanger(
        waveform: Tensor,
        sample_rate: int,
        delay: float = 0.,
        depth: float = 2.,
        regen: float = 0.,
        width: float = 71.,
        speed: float = 0.5,
        phase: float = 25.,
        modulation: str = 'sinusoidal',
        interpolation: str = 'linear'
) -> Tensor:
    r"""Apply a flanger effect to the audio. Similar to SoX implementation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., channel, time)` .
            Max 4 channels allowed
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz)
        delay (float): desired delay in milliseconds(ms)
            Allowed range of values are 0 to 30
        depth (float): desired delay depth in milliseconds(ms)
            Allowed range of values are 0 to 10
        regen (float): desired regen(feeback gain) in dB
            Allowed range of values are -95 to 95
        width (float):  desired width(delay gain) in dB
            Allowed range of values are 0 to 100
        speed (float):  modulation speed in Hz
            Allowed range of values are 0.1 to 10
        phase (float):  percentage phase-shift for multi-channel
            Allowed range of values are 0 to 100
        modulation (str):  Use either "sinusoidal" or "triangular" modulation. (Default: ``sinusoidal``)
        interpolation (str): Use either "linear" or "quadratic" for delay-line interpolation. (Default: ``linear``)

    Returns:
        Tensor: Waveform of dimension of `(..., channel, time)`

    References:
        http://sox.sourceforge.net/sox.html

        Scott Lehman, Effects Explained,
        https://web.archive.org/web/20051125072557/http://www.harmony-central.com/Effects/effects-explained.html
    """

    if modulation not in ('sinusoidal', 'triangular'):
        raise ValueError("Only 'sinusoidal' or 'triangular' modulation allowed")

    if interpolation not in ('linear', 'quadratic'):
        raise ValueError("Only 'linear' or 'quadratic' interpolation allowed")

    actual_shape = waveform.shape
    device, dtype = waveform.device, waveform.dtype

    if actual_shape[-2] > 4:
        raise ValueError("Max 4 channels allowed")

    # convert to 3D (batch, channels, time)
    waveform = waveform.view(-1, actual_shape[-2], actual_shape[-1])

    # Scaling
    feedback_gain = regen / 100
    delay_gain = width / 100
    channel_phase = phase / 100
    delay_min = delay / 1000
    delay_depth = depth / 1000

    n_channels = waveform.shape[-2]

    if modulation == 'sinusoidal':
        wave_type = 'SINE'
    else:
        wave_type = 'TRIANGLE'

    # Balance output:
    in_gain = 1. / (1 + delay_gain)
    delay_gain = delay_gain / (1 + delay_gain)

    # Balance feedback loop:
    delay_gain = delay_gain * (1 - abs(feedback_gain))

    delay_buf_length = int((delay_min + delay_depth) * sample_rate + 0.5)
    delay_buf_length = delay_buf_length + 2

    delay_bufs = torch.zeros(waveform.shape[0], n_channels, delay_buf_length, dtype=dtype, device=device)
    delay_last = torch.zeros(waveform.shape[0], n_channels, dtype=dtype, device=device)

    lfo_length = int(sample_rate / speed)

    table_min = math.floor(delay_min * sample_rate + 0.5)
    table_max = delay_buf_length - 2.

    lfo = _generate_wave_table(wave_type=wave_type,
                               data_type='FLOAT',
                               table_size=lfo_length,
                               min=float(table_min),
                               max=float(table_max),
                               phase=3 * math.pi / 2,
                               device=device)

    output_waveform = torch.zeros_like(waveform, dtype=dtype, device=device)

    delay_buf_pos = 0
    lfo_pos = 0
    channel_idxs = torch.arange(0, n_channels, device=device)

    for i in range(waveform.shape[-1]):

        delay_buf_pos = (delay_buf_pos + delay_buf_length - 1) % delay_buf_length

        cur_channel_phase = (channel_idxs * lfo_length * channel_phase + .5).to(torch.int64)
        delay_tensor = lfo[(lfo_pos + cur_channel_phase) % lfo_length]
        frac_delay = torch.frac(delay_tensor)
        delay_tensor = torch.floor(delay_tensor)

        int_delay = delay_tensor.to(torch.int64)

        temp = waveform[:, :, i]

        delay_bufs[:, :, delay_buf_pos] = temp + delay_last * feedback_gain

        delayed_0 = delay_bufs[:, channel_idxs, (delay_buf_pos + int_delay) % delay_buf_length]

        int_delay = int_delay + 1

        delayed_1 = delay_bufs[:, channel_idxs, (delay_buf_pos + int_delay) % delay_buf_length]

        int_delay = int_delay + 1

        if interpolation == 'linear':
            delayed = delayed_0 + (delayed_1 - delayed_0) * frac_delay
        else:
            delayed_2 = delay_bufs[:, channel_idxs, (delay_buf_pos + int_delay) % delay_buf_length]

            int_delay = int_delay + 1

            delayed_2 = delayed_2 - delayed_0
            delayed_1 = delayed_1 - delayed_0
            a = delayed_2 * .5 - delayed_1
            b = delayed_1 * 2 - delayed_2 * .5

            delayed = delayed_0 + (a * frac_delay + b) * frac_delay

        delay_last = delayed
        output_waveform[:, :, i] = waveform[:, :, i] * in_gain + delayed * delay_gain

        lfo_pos = (lfo_pos + 1) % lfo_length

    return output_waveform.clamp(min=-1, max=1).view(actual_shape)


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


def gain(
        waveform: Tensor,
        gain_db: float = 1.0
) -> Tensor:
    r"""Apply amplification or attenuation to the whole waveform.

    Args:
       waveform (Tensor): Tensor of audio of dimension (..., time).
       gain_db (float, optional) Gain adjustment in decibels (dB) (Default: ``1.0``).

    Returns:
       Tensor: the whole waveform amplified by gain_db.
    """
    if (gain_db == 0):
        return waveform

    ratio = 10 ** (gain_db / 20)

    return waveform * ratio


def _add_noise_shaping(
        dithered_waveform: Tensor,
        waveform: Tensor
) -> Tensor:
    r"""Noise shaping is calculated by error:
    error[n] = dithered[n] - original[n]
    noise_shaped_waveform[n] = dithered[n] + error[n-1]
    """
    wf_shape = waveform.size()
    waveform = waveform.reshape(-1, wf_shape[-1])

    dithered_shape = dithered_waveform.size()
    dithered_waveform = dithered_waveform.reshape(-1, dithered_shape[-1])

    error = dithered_waveform - waveform

    # add error[n-1] to dithered_waveform[n], so offset the error by 1 index
    for index in range(error.size()[0]):
        err = error[index]
        error_offset = torch.cat((torch.zeros(1), err))
        error[index] = error_offset[:waveform.size()[1]]

    noise_shaped = dithered_waveform + error
    return noise_shaped.reshape(dithered_shape[:-1] + noise_shaped.shape[-1:])


def _apply_probability_distribution(
        waveform: Tensor,
        density_function: str = "TPDF"
) -> Tensor:
    r"""Apply a probability distribution function on a waveform.

    Triangular probability density function (TPDF) dither noise has a
    triangular distribution; values in the center of the range have a higher
    probability of occurring.

    Rectangular probability density function (RPDF) dither noise has a
    uniform distribution; any value in the specified range has the same
    probability of occurring.

    Gaussian probability density function (GPDF) has a normal distribution.
    The relationship of probabilities of results follows a bell-shaped,
    or Gaussian curve, typical of dither generated by analog sources.
    Args:
        waveform (Tensor): Tensor of audio of dimension (..., time)
        probability_density_function (str, optional): The density function of a
           continuous random variable (Default: ``"TPDF"``)
           Options: Triangular Probability Density Function - `TPDF`
                    Rectangular Probability Density Function - `RPDF`
                    Gaussian Probability Density Function - `GPDF`
    Returns:
        Tensor: waveform dithered with TPDF
    """

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    channel_size = waveform.size()[0] - 1
    time_size = waveform.size()[-1] - 1

    random_channel = int(torch.randint(channel_size, [1, ]).item()) if channel_size > 0 else 0
    random_time = int(torch.randint(time_size, [1, ]).item()) if time_size > 0 else 0

    number_of_bits = 16
    up_scaling = 2 ** (number_of_bits - 1) - 2
    signal_scaled = waveform * up_scaling
    down_scaling = 2 ** (number_of_bits - 1)

    signal_scaled_dis = waveform
    if (density_function == "RPDF"):
        RPDF = waveform[random_channel][random_time] - 0.5

        signal_scaled_dis = signal_scaled + RPDF
    elif (density_function == "GPDF"):
        # TODO Replace by distribution code once
        # https://github.com/pytorch/pytorch/issues/29843 is resolved
        # gaussian = torch.distributions.normal.Normal(torch.mean(waveform, -1), 1).sample()

        num_rand_variables = 6

        gaussian = waveform[random_channel][random_time]
        for ws in num_rand_variables * [time_size]:
            rand_chan = int(torch.randint(channel_size, [1, ]).item())
            gaussian += waveform[rand_chan][int(torch.randint(ws, [1, ]).item())]

        signal_scaled_dis = signal_scaled + gaussian
    else:
        # dtype needed for https://github.com/pytorch/pytorch/issues/32358
        TPDF = torch.bartlett_window(time_size + 1, dtype=signal_scaled.dtype, device=signal_scaled.device)
        TPDF = TPDF.repeat((channel_size + 1), 1)
        signal_scaled_dis = signal_scaled + TPDF

    quantised_signal_scaled = torch.round(signal_scaled_dis)
    quantised_signal = quantised_signal_scaled / down_scaling

    # unpack batch
    return quantised_signal.reshape(shape[:-1] + quantised_signal.shape[-1:])


def dither(
        waveform: Tensor,
        density_function: str = "TPDF",
        noise_shaping: bool = False
) -> Tensor:
    r"""Dither increases the perceived dynamic range of audio stored at a
    particular bit-depth by eliminating nonlinear truncation distortion
    (i.e. adding minimally perceived noise to mask distortion caused by quantization).
    Args:
       waveform (Tensor): Tensor of audio of dimension (..., time)
       density_function (str, optional): The density function of a continuous random variable (Default: ``"TPDF"``)
           Options: Triangular Probability Density Function - `TPDF`
                    Rectangular Probability Density Function - `RPDF`
                    Gaussian Probability Density Function - `GPDF`
       noise_shaping (bool, optional): a filtering process that shapes the spectral
           energy of quantisation error (Default: ``False``)

    Returns:
       Tensor: waveform dithered
    """
    dithered = _apply_probability_distribution(waveform, density_function=density_function)

    if noise_shaping:
        return _add_noise_shaping(dithered, waveform)
    else:
        return dithered


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


def _measure(
    measure_len_ws: int,
    samples: Tensor,
    spectrum: Tensor,
    noise_spectrum: Tensor,
    spectrum_window: Tensor,
    spectrum_start: int,
    spectrum_end: int,
    cepstrum_window: Tensor,
    cepstrum_start: int,
    cepstrum_end: int,
    noise_reduction_amount: float,
    measure_smooth_time_mult: float,
    noise_up_time_mult: float,
    noise_down_time_mult: float,
    index_ns: int,
    boot_count: int
) -> float:

    assert spectrum.size()[-1] == noise_spectrum.size()[-1]

    samplesLen_ns = samples.size()[-1]
    dft_len_ws = spectrum.size()[-1]

    dftBuf = torch.zeros(dft_len_ws)

    _index_ns = torch.tensor([index_ns] + [
        (index_ns + i) % samplesLen_ns
        for i in range(1, measure_len_ws)
    ])
    dftBuf[:measure_len_ws] = \
        samples[_index_ns] * spectrum_window[:measure_len_ws]

    # memset(c->dftBuf + i, 0, (p->dft_len_ws - i) * sizeof(*c->dftBuf));
    dftBuf[measure_len_ws:dft_len_ws].zero_()

    # lsx_safe_rdft((int)p->dft_len_ws, 1, c->dftBuf);
    _dftBuf = torch.rfft(dftBuf, 1)

    # memset(c->dftBuf, 0, p->spectrum_start * sizeof(*c->dftBuf));
    _dftBuf[:spectrum_start].zero_()

    mult: float = boot_count / (1. + boot_count) \
        if boot_count >= 0 \
        else measure_smooth_time_mult

    _d = complex_norm(_dftBuf[spectrum_start:spectrum_end])
    spectrum[spectrum_start:spectrum_end].mul_(mult).add_(_d * (1 - mult))
    _d = spectrum[spectrum_start:spectrum_end] ** 2

    _zeros = torch.zeros(spectrum_end - spectrum_start)
    _mult = _zeros \
        if boot_count >= 0 \
        else torch.where(
            _d > noise_spectrum[spectrum_start:spectrum_end],
            torch.tensor(noise_up_time_mult),   # if
            torch.tensor(noise_down_time_mult)  # else
        )

    noise_spectrum[spectrum_start:spectrum_end].mul_(_mult).add_(_d * (1 - _mult))
    _d = torch.sqrt(
        torch.max(
            _zeros,
            _d - noise_reduction_amount * noise_spectrum[spectrum_start:spectrum_end]))

    _cepstrum_Buf: Tensor = torch.zeros(dft_len_ws >> 1)
    _cepstrum_Buf[spectrum_start:spectrum_end] = _d * cepstrum_window
    _cepstrum_Buf[spectrum_end:dft_len_ws >> 1].zero_()

    # lsx_safe_rdft((int)p->dft_len_ws >> 1, 1, c->dftBuf);
    _cepstrum_Buf = torch.rfft(_cepstrum_Buf, 1)

    result: float = float(torch.sum(
        complex_norm(
            _cepstrum_Buf[cepstrum_start:cepstrum_end],
            power=2.0)))
    result = \
        math.log(result / (cepstrum_end - cepstrum_start)) \
        if result > 0 \
        else -math.inf
    return max(0, 21 + result)


def vad(
    waveform: Tensor,
    sample_rate: int,
    trigger_level: float = 7.0,
    trigger_time: float = 0.25,
    search_time: float = 1.0,
    allowed_gap: float = 0.25,
    pre_trigger_time: float = 0.0,
    # Fine-tuning parameters
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
    lp_lifter_freq: float = 2000.,
) -> Tensor:
    r"""Voice Activity Detector. Similar to SoX implementation.
    Attempts to trim silence and quiet background sounds from the ends of recordings of speech.
    The algorithm currently uses a simple cepstral power measurement to detect voice,
    so may be fooled by other things, especially music.

    The effect can trim only from the front of the audio,
    so in order to trim from the back, the reverse effect must also be used.

    Args:
        waveform (Tensor): Tensor of audio of dimension `(..., time)`
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

    Returns:
        Tensor: Tensor of audio of dimension (..., time).

    References:
        http://sox.sourceforge.net/sox.html
    """

    measure_duration: float = 2.0 / measure_freq \
        if measure_duration is None \
        else measure_duration

    measure_len_ws = int(sample_rate * measure_duration + .5)
    measure_len_ns = measure_len_ws
    # for (dft_len_ws = 16; dft_len_ws < measure_len_ws; dft_len_ws <<= 1);
    dft_len_ws = 16
    while (dft_len_ws < measure_len_ws):
        dft_len_ws *= 2

    measure_period_ns = int(sample_rate / measure_freq + .5)
    measures_len = math.ceil(search_time * measure_freq)
    search_pre_trigger_len_ns = measures_len * measure_period_ns
    gap_len = int(allowed_gap * measure_freq + .5)

    fixed_pre_trigger_len_ns = int(pre_trigger_time * sample_rate + .5)
    samplesLen_ns = fixed_pre_trigger_len_ns + search_pre_trigger_len_ns + measure_len_ns

    spectrum_window = torch.zeros(measure_len_ws)
    for i in range(measure_len_ws):
        # sox.h:741 define SOX_SAMPLE_MIN (sox_sample_t)SOX_INT_MIN(32)
        spectrum_window[i] = 2. / math.sqrt(float(measure_len_ws))
    # lsx_apply_hann(spectrum_window, (int)measure_len_ws);
    spectrum_window *= torch.hann_window(measure_len_ws, dtype=torch.float)

    spectrum_start: int = int(hp_filter_freq / sample_rate * dft_len_ws + .5)
    spectrum_start: int = max(spectrum_start, 1)
    spectrum_end: int = int(lp_filter_freq / sample_rate * dft_len_ws + .5)
    spectrum_end: int = min(spectrum_end, dft_len_ws // 2)

    cepstrum_window = torch.zeros(spectrum_end - spectrum_start)
    for i in range(spectrum_end - spectrum_start):
        cepstrum_window[i] = 2. / math.sqrt(float(spectrum_end) - spectrum_start)
    # lsx_apply_hann(cepstrum_window,(int)(spectrum_end - spectrum_start));
    cepstrum_window *= torch.hann_window(spectrum_end - spectrum_start, dtype=torch.float)

    cepstrum_start = math.ceil(sample_rate * .5 / lp_lifter_freq)
    cepstrum_end = math.floor(sample_rate * .5 / hp_lifter_freq)
    cepstrum_end = min(cepstrum_end, dft_len_ws // 4)

    assert cepstrum_end > cepstrum_start

    noise_up_time_mult = math.exp(-1. / (noise_up_time * measure_freq))
    noise_down_time_mult = math.exp(-1. / (noise_down_time * measure_freq))
    measure_smooth_time_mult = math.exp(-1. / (measure_smooth_time * measure_freq))
    trigger_meas_time_mult = math.exp(-1. / (trigger_time * measure_freq))

    boot_count_max = int(boot_time * measure_freq - .5)
    measure_timer_ns = measure_len_ns
    boot_count = measures_index = flushedLen_ns = samplesIndex_ns = 0

    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

    n_channels, ilen = waveform.size()

    mean_meas = torch.zeros(n_channels)
    samples = torch.zeros(n_channels, samplesLen_ns)
    spectrum = torch.zeros(n_channels, dft_len_ws)
    noise_spectrum = torch.zeros(n_channels, dft_len_ws)
    measures = torch.zeros(n_channels, measures_len)

    has_triggered: bool = False
    num_measures_to_flush: int = 0
    pos: int = 0

    while (pos < ilen and not has_triggered):
        measure_timer_ns -= 1
        for i in range(n_channels):
            samples[i, samplesIndex_ns] = waveform[i, pos]
            # if (!p->measure_timer_ns) {
            if (measure_timer_ns == 0):
                index_ns: int = \
                    (samplesIndex_ns + samplesLen_ns - measure_len_ns) % samplesLen_ns
                meas: float = _measure(
                    measure_len_ws=measure_len_ws,
                    samples=samples[i],
                    spectrum=spectrum[i],
                    noise_spectrum=noise_spectrum[i],
                    spectrum_window=spectrum_window,
                    spectrum_start=spectrum_start,
                    spectrum_end=spectrum_end,
                    cepstrum_window=cepstrum_window,
                    cepstrum_start=cepstrum_start,
                    cepstrum_end=cepstrum_end,
                    noise_reduction_amount=noise_reduction_amount,
                    measure_smooth_time_mult=measure_smooth_time_mult,
                    noise_up_time_mult=noise_up_time_mult,
                    noise_down_time_mult=noise_down_time_mult,
                    index_ns=index_ns,
                    boot_count=boot_count)
                measures[i, measures_index] = meas
                mean_meas[i] = mean_meas[i] * trigger_meas_time_mult + meas * (1. - trigger_meas_time_mult)

                has_triggered = has_triggered or (mean_meas[i] >= trigger_level)
                if has_triggered:
                    n: int = measures_len
                    k: int = measures_index
                    jTrigger: int = n
                    jZero: int = n
                    j: int = 0

                    for j in range(n):
                        if (measures[i, k] >= trigger_level) and (j <= jTrigger + gap_len):
                            jZero = jTrigger = j
                        elif (measures[i, k] == 0) and (jTrigger >= jZero):
                            jZero = j
                        k = (k + n - 1) % n
                    j = min(j, jZero)
                    # num_measures_to_flush = range_limit(j, num_measures_to_flush, n);
                    num_measures_to_flush = (min(max(num_measures_to_flush, j), n))
                # end if has_triggered
            # end if (measure_timer_ns == 0):
        # end for
        samplesIndex_ns += 1
        pos += 1
    # end while
        if samplesIndex_ns == samplesLen_ns:
            samplesIndex_ns = 0
        if measure_timer_ns == 0:
            measure_timer_ns = measure_period_ns
            measures_index += 1
            measures_index = measures_index % measures_len
            if boot_count >= 0:
                boot_count = -1 if boot_count == boot_count_max else boot_count + 1

        if has_triggered:
            flushedLen_ns = (measures_len - num_measures_to_flush) * measure_period_ns
            samplesIndex_ns = (samplesIndex_ns + flushedLen_ns) % samplesLen_ns

    res = waveform[:, pos - samplesLen_ns + flushedLen_ns:]
    # unpack batch
    return res.view(shape[:-1] + res.shape[-1:])
