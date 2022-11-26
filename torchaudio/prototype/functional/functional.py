import math
import warnings

import torch

from torchaudio.functional.functional import _create_triangular_filterbank


def _check_shape_compatible(x: torch.Tensor, y: torch.Tensor, allow_broadcast: bool) -> None:
    if x.ndim != y.ndim:
        raise ValueError(f"The operands must be the same dimension (got {x.ndim} and {y.ndim}).")
    if not allow_broadcast:
        if x.shape[:-1] != y.shape[:-1]:
            raise ValueError(f"Leading dimensions of x and y don't match (got {x.shape} and {y.shape}).")
    else:
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
            (leading dimensions must be broadcast-able to those of ``x``).
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
    _check_shape_compatible(x, y, allow_broadcast=True)
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
            (leading dimensions must match those of ``x``).
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
    _check_shape_compatible(x, y, allow_broadcast=False)
    _check_convolve_mode(mode)

    x_size, y_size = x.size(-1), y.size(-1)

    if x.size(-1) < y.size(-1):
        x, y = y, x

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


def add_noise(waveform: torch.Tensor, noise: torch.Tensor, lengths: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
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
        lengths (torch.Tensor): Valid lengths of signals in ``waveform`` and ``noise``, with shape `(...,)`
            (leading dimensions must match those of ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    """

    if not (waveform.ndim - 1 == noise.ndim - 1 == lengths.ndim == snr.ndim):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    # compute scale
    mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
        -1
    )  # (*, L) < (*, 1) = (*, L)
    energy_signal = torch.linalg.vector_norm(waveform * mask, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(noise * mask, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)


def _hz_to_bark(freqs: float, bark_scale: str = "traunmuller") -> float:
    r"""Convert Hz to Barks.

    Args:
        freqs (float): Frequencies in Hz
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        barks (float): Frequency in Barks
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError('bark_scale should be one of "schroeder", "traunmuller" or "wang".')

    if bark_scale == "wang":
        return 6.0 * math.asinh(freqs / 600.0)
    elif bark_scale == "schroeder":
        return 7.0 * math.asinh(freqs / 650.0)
    # Traunmuller Bark scale
    barks = ((26.81 * freqs) / (1960.0 + freqs)) - 0.53
    # Bark value correction
    if barks < 2:
        barks += 0.15 * (2 - barks)
    elif barks > 20.1:
        barks += 0.22 * (barks - 20.1)

    return barks


def _bark_to_hz(barks: torch.Tensor, bark_scale: str = "traunmuller") -> torch.Tensor:
    """Convert bark bin numbers to frequencies.

    Args:
        barks (torch.Tensor): Bark frequencies
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        freqs (torch.Tensor): Barks converted in Hz
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError('bark_scale should be one of "traunmuller", "schroeder" or "wang".')

    if bark_scale == "wang":
        return 600.0 * torch.sinh(barks / 6.0)
    elif bark_scale == "schroeder":
        return 650.0 * torch.sinh(barks / 7.0)
    # Bark value correction
    if any(barks < 2):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif any(barks > 20.1):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22

    # Traunmuller Bark scale
    freqs = 1960 * ((barks + 0.53) / (26.28 - barks))

    return freqs


def barkscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_barks: int,
    sample_rate: int,
    bark_scale: str = "traunmuller",
) -> torch.Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    .. image:: https://download.pytorch.org/torchaudio/doc-assets/bark_fbanks.png
        :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_barks (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        torch.Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_barks``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * barkscale_fbanks(A.size(-1), ...)``.

    """

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate bark freq bins
    m_min = _hz_to_bark(f_min, bark_scale=bark_scale)
    m_max = _hz_to_bark(f_max, bark_scale=bark_scale)

    m_pts = torch.linspace(m_min, m_max, n_barks + 2)
    f_pts = _bark_to_hz(m_pts, bark_scale=bark_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one bark filterbank has all zero values. "
            f"The value for `n_barks` ({n_barks}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb
