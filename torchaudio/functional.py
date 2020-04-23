# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

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
    "deemph_biquad",
    "riaa_biquad",
    "biquad",
    "contrast",
    "dcshift",
    "overdrive",
    'mask_along_axis',
    'mask_along_axis_iid',
    'sliding_window_cmn',
    "vad",
]


def istft(
        stft_matrix: Tensor,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[Tensor] = None,
        center: bool = True,
        pad_mode: str = "reflect",
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
        pad_mode (str, optional): Controls the padding method used when ``center`` is True. (Default:
            ``"reflect"``)
        normalized (bool, optional): Whether the STFT was normalized. (Default: ``False``)
        onesided (bool, optional): Whether the STFT is onesided. (Default: ``True``)
        length (int or None, optional): The amount to trim the signal by (i.e. the
            original signal length). (Default: whole signal)

    Returns:
        Tensor: Least squares estimation of the original signal of size (..., signal_length)
    """
    stft_matrix_dim = stft_matrix.dim()
    assert 3 <= stft_matrix_dim, "Incorrect stft dimension: %d" % (stft_matrix_dim)
    assert stft_matrix.numel() > 0

    if stft_matrix_dim == 3:
        # add a channel dimension
        stft_matrix = stft_matrix.unsqueeze(0)

    # pack batch
    shape = stft_matrix.size()
    stft_matrix = stft_matrix.view(-1, shape[-3], shape[-2], shape[-1])

    dtype = stft_matrix.dtype
    device = stft_matrix.device
    fft_size = stft_matrix.size(1)
    assert (onesided and n_fft // 2 + 1 == fft_size) or (
        not onesided and n_fft == fft_size
    ), (
        "one_sided implies that n_fft // 2 + 1 == fft_size and not one_sided implies n_fft == fft_size. "
        + "Given values were onesided: %s, n_fft: %d, fft_size: %d"
        % ("True" if onesided else False, n_fft, fft_size)
    )

    # use stft defaults for Optionals
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # There must be overlap
    assert 0 < hop_length <= win_length
    assert 0 < win_length <= n_fft

    if window is None:
        window = torch.ones(win_length, device=device, dtype=dtype)

    assert window.dim() == 1 and window.size(0) == win_length

    if win_length != n_fft:
        # center window with pad left and right zeros
        left = (n_fft - win_length) // 2
        window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        assert window.size(0) == n_fft
    # win_length and n_fft are synonymous from here on

    stft_matrix = stft_matrix.transpose(1, 2)  # size (channel, n_frame, fft_size, 2)
    stft_matrix = torch.irfft(
        stft_matrix, 1, normalized, onesided, signal_sizes=(n_fft,)
    )  # size (channel, n_frame, n_fft)

    assert stft_matrix.size(2) == n_fft
    n_frame = stft_matrix.size(1)

    ytmp = stft_matrix * window.view(1, 1, n_fft)  # size (channel, n_frame, n_fft)
    # each column of a channel is a frame which needs to be overlap added at the right place
    ytmp = ytmp.transpose(1, 2)  # size (channel, n_fft, n_frame)

    # this does overlap add where the frames of ytmp are added such that the i'th frame of
    # ytmp is added starting at i*hop_length in the output
    y = torch.nn.functional.fold(
        ytmp, (1, (n_frame - 1) * hop_length + n_fft), (1, n_fft), stride=(1, hop_length)
    ).squeeze(2)

    # do the same for the window function
    window_sq = (
        window.pow(2).view(n_fft, 1).repeat((1, n_frame)).unsqueeze(0)
    )  # size (1, n_fft, n_frame)
    window_envelop = torch.nn.functional.fold(
        window_sq, (1, (n_frame - 1) * hop_length + n_fft), (1, n_fft), stride=(1, hop_length)
    ).squeeze(2)  # size (1, 1, expected_signal_len)

    expected_signal_len = n_fft + hop_length * (n_frame - 1)
    assert y.size(2) == expected_signal_len
    assert window_envelop.size(2) == expected_signal_len

    half_n_fft = n_fft // 2
    # we need to trim the front padding away if center
    start = half_n_fft if center else 0
    end = -half_n_fft if length is None else start + length

    y = y[:, :, start:end]
    window_envelop = window_envelop[:, :, start:end]

    # check NOLA non-zero overlap condition
    window_envelop_lowest = window_envelop.abs().min()
    assert window_envelop_lowest > 1e-11, "window overlap add min: %f" % (
        window_envelop_lowest
    )

    y = (y / window_envelop).squeeze(1)  # size (channel, expected_signal_len)

    # unpack batch
    y = y.view(shape[:-3] + y.shape[-1:])

    if stft_matrix_dim == 3:  # remove the channel dimension
        y = y.squeeze(0)

    return y


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
    waveform = waveform.view(-1, shape[-1])

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = torch.stft(
        waveform, n_fft, hop_length, win_length, window, True, "reflect", False, True
    )

    # unpack batch
    spec_f = spec_f.view(shape[:-1] + spec_f.shape[-3:])

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
    assert momentum < 1, 'momentum=%s > 1 can be unstable' % momentum
    assert momentum > 0, 'momentum=%s < 0' % momentum

    # pack batch
    shape = specgram.size()
    specgram = specgram.view([-1] + list(shape[-2:]))

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
        angles = rebuilt - tprev.mul_(momentum / (1 + momentum))
        angles = angles.div_(complex_norm(angles).add_(1e-16).unsqueeze(-1).expand_as(angles))

    # Return the final phase estimates
    waveform = istft(specgram * angles,
                     n_fft=n_fft,
                     hop_length=hop_length,
                     win_length=win_length,
                     window=window,
                     length=length)

    # unpack batch
    waveform = waveform.view(shape[:-2] + waveform.shape[-1:])

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
        sample_rate: int
) -> Tensor:
    r"""Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * create_fb_matrix(A.size(-1), ...)``.
    """
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
    if power == 1.0:
        return torch.norm(complex_tensor, 2, -1)
    return torch.norm(complex_tensor, 2, -1).pow(power)


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
    complex_specgrams = complex_specgrams.view([-1] + list(shape[-3:]))

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
    complex_specgrams_stretch = complex_specgrams_stretch.view(shape[:-3] + complex_specgrams_stretch.shape[1:])

    return complex_specgrams_stretch


def lfilter(
        waveform: Tensor,
        a_coeffs: Tensor,
        b_coeffs: Tensor
) -> Tensor:
    r"""Perform an IIR filter by evaluating difference equation.

    Args:
        waveform (Tensor): audio waveform of dimension of `(..., time)`.  Must be normalized to -1 to 1.
        a_coeffs (Tensor): denominator coefficients of difference equation of dimension of `(n_order + 1)`.
                                Lower delays coefficients are first, e.g. `[a0, a1, a2, ...]`.
                                Must be same size as b_coeffs (pad with 0's as necessary).
        b_coeffs (Tensor): numerator coefficients of difference equation of dimension of `(n_order + 1)`.
                                 Lower delays coefficients are first, e.g. `[b0, b1, b2, ...]`.
                                 Must be same size as a_coeffs (pad with 0's as necessary).

    Returns:
        Tensor: Waveform with dimension of `(..., time)`.  Output will be clipped to -1 to 1.
    """
    # pack batch
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])

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

    output = torch.clamp(padded_output_waveform[:, (n_order - 1):], min=-1., max=1.)

    # unpack batch
    output = output.view(shape[:-1] + output.shape[-1:])

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


def mask_along_axis_iid(
        specgrams: Tensor,
        mask_param: int,
        mask_value: float,
        axis: int
) -> Tensor:
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

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
    specgram = specgram.view([-1] + list(shape[-2:]))

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
    specgram = specgram.view(shape[:-2] + specgram.shape[-2:])

    return specgram


def compute_deltas(
        specgram: Tensor,
        win_length: int = 5,
        mode: str = "replicate"
) -> Tensor:
    r"""Compute delta coefficients of a tensor, usually a spectrogram:

    .. math::
        d_t = \frac{\sum_{n=1}^{\text{N}} n (c_{t+n} - c_{t-n})}{2 \sum_{n=1}^{\text{N} n^2}

    where :math:`d_t` is the deltas at time :math:`t`,
    :math:`c_t` is the spectrogram coeffcients at time :math:`t`,
    :math:`N` is (`win_length`-1)//2.

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
    specgram = specgram.view(1, -1, shape[-1])

    assert win_length >= 3

    n = (win_length - 1) // 2

    # twice sum of integer squared
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(specgram.shape[1], 1, 1)

    output = torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom

    # unpack batch
    output = output.view(shape)

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
    waveform = waveform.view(-1, waveform.size()[-1])

    dithered_shape = dithered_waveform.size()
    dithered_waveform = dithered_waveform.view(-1, dithered_shape[-1])

    error = dithered_waveform - waveform

    # add error[n-1] to dithered_waveform[n], so offset the error by 1 index
    for index in range(error.size()[0]):
        err = error[index]
        error_offset = torch.cat((torch.zeros(1), err))
        error[index] = error_offset[:waveform.size()[1]]

    noise_shaped = dithered_waveform + error
    return noise_shaped.view(dithered_shape[:-1] + noise_shaped.shape[-1:])


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
    waveform = waveform.view(-1, shape[-1])

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
    return quantised_signal.view(shape[:-1] + quantised_signal.shape[-1:])


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
    waveform = waveform.view([-1] + shape[-1:])

    nccf = _compute_nccf(waveform, sample_rate, frame_time, freq_low)
    indices = _find_max_per_frame(nccf, sample_rate, freq_high)
    indices = _median_smoothing(indices, win_length)

    # Convert indices to frequency
    EPSILON = 10 ** (-9)
    freq = sample_rate / (EPSILON + indices.to(torch.float))

    # unpack batch
    freq = freq.view(shape[:-1] + list(freq.shape[-1:]))

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
