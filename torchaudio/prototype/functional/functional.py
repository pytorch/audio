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
        x (torch.Tensor): First convolution operand, with shape `(*, N)`.
        y (torch.Tensor): Second convolution operand, with shape `(*, M)`
            (leading dimensions must match those of ``x``).

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(*, N + M - 1)`, where
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
        x (torch.Tensor): First convolution operand, with shape `(*, N)`.
        y (torch.Tensor): Second convolution operand, with shape `(*, M)`
            (leading dimensions must match those of ``x``).

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(*, N + M - 1)`, where
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
    r"""Scales and adds noise from multiple sources to waveform according to signal-to-noise ratios.

    Specifically, for each waveform vector :math:`x \in \mathbb{R}^L` and noise vectors
    :math:`n_1, \ldots, n_N \in \mathbb{R}^L` corresponding to :math:`N` sources, the
    function computes output :math:`y` as

    .. math::
        y = x + \sum_{i = 1}^N a_i n_i

    , where

    .. math::
        a_i = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n_i||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}_i}{10}} }

    , with :math:`\text{SNR}_i` being the desired signal-to-noise ratio between :math:`x` and :math:`n_i`, in dB.

    Args:
        waveform (torch.Tensor): (*, L)
        noise (torch.Tensor): (*, N, L)
        lengths (torch.Tensor): (*,); actual lengths of signals in `waveform`.
        snr (torch.Tensor): (*, N); in dB.

    Returns:
        torch.Tensor: (*, length)
    """
    # compute scale
    mask = torch.arange(0, waveform.size(-1)).expand(waveform.shape) < lengths.unsqueeze(-1)  # (*, L) < (*, 1) = (*, L)
    energy_signal = torch.linalg.vector_norm(waveform * mask, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(noise * mask.unsqueeze(-2), ord=2, dim=-1) ** 2  # (*, N)
    original_snr = energy_signal.unsqueeze(-1) / energy_noise  # (*, N)
    snr_abs = 10 ** (snr / 10.0)  # (*, N)
    scale = (original_snr / snr_abs).sqrt()  # (*, N)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, N, 1) * (*, N, L) = (*, N, L)

    # sum-reduce scaled noise
    scaled_noise = scaled_noise.sum(-2)  # (*, L)

    return waveform + scaled_noise  # (*, L)
