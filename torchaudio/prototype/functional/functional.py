import math

import torch


def convolve(x: torch.Tensor, y: torch.Tensor, use_fft: bool = False) -> torch.Tensor:
    """
    Convolves inputs along last dimension. Note that, in contrast to ``torch.nn.functional.conv1d``, this
    function applies the true `convolution`_ operator.

    Args:
        x (torch.Tensor): First convolution operand, with shape `(*, N)`.
        y (torch.Tensor): Second convolution operand, with shape `(*, M)`
            (leading dimensions must match those of ``x``).
        use_fft (bool, optional): If ``True``, then convolve ``x`` and ``y`` using FFT.
            Otherwise, convolve ``x`` and ``y`` directly via sums. (Default: ``False``)

    Returns:
        torch.Tensor: Result of convolving ``x`` and ``y``, with shape `(*, N + M - 1)`, where
        the leading dimensions match those of ``x``.

    .. _convolution:
        https://en.wikipedia.org/wiki/Convolution
    """
    if use_fft:
        n = x.size(-1) + y.size(-1) - 1
        fresult = torch.fft.rfft(x, n=n) * torch.fft.rfft(y, n=n)
        output = torch.fft.irfft(fresult, n=n)
    else:
        num_signals = math.prod(dim for dim in x.shape[:-1])
        reshaped_x = x.reshape((num_signals, x.size(-1)))
        reshaped_y = y.reshape((num_signals, y.size(-1)))
        output = torch.nn.functional.conv1d(
            input=reshaped_x,
            weight=reshaped_y.flip(-1).unsqueeze(1),
            stride=1,
            groups=reshaped_x.size(0),
            padding=reshaped_y.size(-1) - 1,
        )
        output_shape = list(x.shape[:-1]) + [-1]
        output = output.reshape(output_shape)
    return output
