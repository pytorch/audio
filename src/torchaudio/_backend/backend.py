import os
from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, Tuple, Union

from torch import Tensor
from torchaudio.io import CodecConfig

from .common import AudioMetaData


class Backend(ABC):
    @staticmethod
    @abstractmethod
    def info(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], buffer_size: int = 4096) -> AudioMetaData:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(
        uri: Union[BinaryIO, str, os.PathLike],
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ) -> Tuple[Tensor, int]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save(
        uri: Union[BinaryIO, str, os.PathLike],
        src: Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        compression: Optional[Union[CodecConfig, float, int]] = None,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def can_decode(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str]) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def can_encode(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str]) -> bool:
        raise NotImplementedError
