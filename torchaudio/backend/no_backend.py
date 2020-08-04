from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

from torch import Tensor

from . import common
from .common import SignalInfo, EncodingInfo


@common._impl_load
def load(
    filepath: Union[str, Path],
    out: Optional[Tensor] = None,
    normalization: Union[bool, float, Callable] = True,
    channels_first: bool = True,
    num_frames: int = 0,
    offset: int = 0,
    signalinfo: Optional[SignalInfo] = None,
    encodinginfo: Optional[EncodingInfo] = None,
    filetype: Optional[str] = None,
) -> Tuple[Tensor, int]:
    raise RuntimeError("No audio I/O backend is available.")


@common._impl_load_wav
def load_wav(filepath: Union[str, Path], **kwargs: Any) -> Tuple[Tensor, int]:
    raise RuntimeError("No audio I/O backend is available.")


@common._impl_save
def save(
    filepath: str,
    src: Tensor,
    sample_rate: int,
    precision: int = 16,
    channels_first: bool = True,
) -> None:
    raise RuntimeError("No audio I/O backend is available.")


@common._impl_info
def info(filepath: str) -> Tuple[SignalInfo, EncodingInfo]:
    raise RuntimeError("No audio I/O backend is available.")
