import torch


def _check_convolve_inputs(x: torch.Tensor, y: torch.Tensor) -> None:
    if x.shape[:-1] != y.shape[:-1]:
        raise ValueError(f"Leading dimensions of x and y don't match (got {x.shape} and {y.shape}).")


def fftconvolve(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    Convolves inputs along their last dimension using FFT.
    Note that, in contrast to ``torch.nn.functional.conv1d``, which actually applies the valid cross-correlation
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
    Note that, in contrast to ``torch.nn.functional.conv1d``, which actually applies the valid cross-correlation
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
