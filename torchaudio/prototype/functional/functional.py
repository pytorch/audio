import torch


def _check_convolve_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError(f"Leading dimensions of x and y don't match (got {x.shape} and {y.shape}).")


def fftconvolve(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
            (leading dimensions must match those of ``x``).

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., N + M - 1)`, where
        the leading dimensions match those of ``x``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """
    _check_convolve_inputs(x, y)

    n = x.size(-1) + y.size(-1) - 1
    fresult = torch.fft.rfft(x, n=n) * torch.fft.rfft(y, n=n)
    return torch.fft.irfft(fresult, n=n)


def convolve(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(..., N + M - 1)`, where
        the leading dimensions match those of ``x``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """
    _check_convolve_inputs(x, y)

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
    return output.reshape(output_shape)


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
