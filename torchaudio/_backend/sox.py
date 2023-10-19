import os
from typing import BinaryIO, Optional, Tuple, Union

import torch
from torchaudio.io import CodecConfig

from .backend import Backend
from .common import AudioMetaData


class SoXBackend(Backend):
    @staticmethod
    def info(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], buffer_size: int = 4096) -> AudioMetaData:
        if hasattr(uri, "read"):
            raise ValueError(
                "SoX backend does not support reading from file-like objects. ",
                "Please use an alternative backend that does support reading from file-like objects, e.g. FFmpeg.",
            )
        else:
            sinfo = torch.ops.torchaudio.sox_io_get_info(uri, format)
            if sinfo:
                return AudioMetaData(*sinfo)
            else:
                raise RuntimeError(f"Failed to fetch metadata for {uri}.")

    @staticmethod
    def load(
        uri: Union[BinaryIO, str, os.PathLike],
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ) -> Tuple[torch.Tensor, int]:
        if hasattr(uri, "read"):
            raise ValueError(
                "SoX backend does not support loading from file-like objects. ",
                "Please use an alternative backend that does support loading from file-like objects, e.g. FFmpeg.",
            )
        else:
            ret = torch.ops.torchaudio.sox_io_load_audio_file(
                uri, frame_offset, num_frames, normalize, channels_first, format
            )
            if not ret:
                raise RuntimeError(f"Failed to load audio from {uri}.")
            return ret

    @staticmethod
    def save(
        uri: Union[BinaryIO, str, os.PathLike],
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        compression: Optional[Union[CodecConfig, float, int]] = None,
    ) -> None:
        if not isinstance(compression, (float, int, type(None))):
            raise ValueError(
                "SoX backend expects non-`None` value for argument `compression` to be of ",
                f"type `float` or `int`, but received value of type {type(compression)}",
            )
        if hasattr(uri, "write"):
            raise ValueError(
                "SoX backend does not support writing to file-like objects. ",
                "Please use an alternative backend that does support writing to file-like objects, e.g. FFmpeg.",
            )
        else:
            torch.ops.torchaudio.sox_io_save_audio_file(
                uri,
                src,
                sample_rate,
                channels_first,
                compression,
                format,
                encoding,
                bits_per_sample,
            )

    @staticmethod
    def can_decode(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str]) -> bool:
        # i.e. not a file-like object.
        return not hasattr(uri, "read")

    @staticmethod
    def can_encode(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str]) -> bool:
        # i.e. not a file-like object.
        return not hasattr(uri, "write")
