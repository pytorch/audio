from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from torch import Tensor


def load(
    filepath: Union[str, Path],
    out: Optional[Tensor] = None,
    normalization: Union[bool, float, Callable] = True,
    channels_first: bool = True,
    num_frames: int = 0,
    offset: int = 0,
    filetype: Optional[str] = None,
) -> Tuple[Tensor, int]:
    raise RuntimeError("No audio I/O backend is available.")


def save(filepath: str, src: Tensor, sample_rate: int, precision: int = 16, channels_first: bool = True) -> None:
    raise RuntimeError("No audio I/O backend is available.")


def info(filepath: str) -> None:
    raise RuntimeError("No audio I/O backend is available.")
