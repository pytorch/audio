"""Compatibility module for fft-related functions

In PyTorch 1.7, the new `torch.fft` module was introduced.

To use this new module, one has to explicitly import `torch.fft`. however this will change
the reference `torch.fft` is pointing from function to module.
And this change takes effect not only in the client code but also in already-imported libraries too.
Similarly, if a library does the explicit import, the rest of the application code must use the
`torch.fft.fft` function.

For this reason, to migrate the deprecated functions of fft-family, we need to use the new
implementation under `torch.fft` without explicitly importing `torch.fft` module.

This module provides a simple interface for the migration, abstracting away
the access to the underlying C functions.

Once the deprecated functions are removed from PyTorch and `torch.fft` starts to always represent
the new module, we can get rid of this module and call functions under `torch.fft` directly.
"""
from typing import Optional

import torch


def rfft(input: torch.Tensor, n: Optional[int] = None, dim: int = -1, norm: Optional[str] = None) -> torch.Tensor:
    # see: https://pytorch.org/docs/master/fft.html#torch.fft.rfft
    return torch._C._fft.fft_rfft(input, n, dim, norm)
