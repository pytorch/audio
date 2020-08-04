from typing import Union, Callable

import torch
from torch import Tensor


def normalize_audio(
    signal: Tensor, normalization: Union[bool, float, Callable]
) -> None:
    """Audio normalization of a tensor in-place.  The normalization can be a bool,
    a number, or a callable that takes the audio tensor as an input. SoX uses
    32-bit signed integers internally, thus bool normalizes based on that assumption.
    """

    if not normalization:
        return

    if isinstance(normalization, bool):
        normalization = 1 << 31

    if isinstance(normalization, (float, int)):
        # normalize with custom value
        signal /= normalization
    elif callable(normalization):
        signal /= normalization(signal)


def check_input(src: Tensor) -> None:
    if not torch.is_tensor(src):
        raise TypeError("Expected a tensor, got %s" % type(src))
    if src.is_cuda:
        raise TypeError("Expected a CPU based tensor, got %s" % type(src))
