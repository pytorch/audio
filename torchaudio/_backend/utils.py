import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import BinaryIO, Dict, Optional, Tuple, Union

import torch
import torchaudio.backend.soundfile_backend as soundfile_backend
from torchaudio._extension import _FFMPEG_INITIALIZED, _SOX_INITIALIZED
from torchaudio.backend.common import AudioMetaData

if _FFMPEG_INITIALIZED:
    from torchaudio.io._compat import info_audio, info_audio_fileobj, load_audio, load_audio_fileobj, save_audio


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
    ) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def save(
        uri: Union[BinaryIO, str, os.PathLike],
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
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


def _map_encoding(encoding: str) -> str:
    for dst in ["PCM_S", "PCM_U", "PCM_F"]:
        if dst in encoding:
            return dst
    if encoding == "PCM_MULAW":
        return "ULAW"
    elif encoding == "PCM_ALAW":
        return "ALAW"
    return encoding


def _get_bits_per_sample(encoding: str, bits_per_sample: int) -> str:
    if m := re.search(r"PCM_\w(\d+)\w*", encoding):
        return int(m.group(1))
    elif encoding in ["PCM_ALAW", "PCM_MULAW"]:
        return 8
    return bits_per_sample


class FFmpegBackend(Backend):
    @staticmethod
    def info(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], buffer_size: int = 4096) -> AudioMetaData:
        if hasattr(uri, "read"):
            metadata = info_audio_fileobj(uri, format, buffer_size=buffer_size)
        else:
            metadata = info_audio(uri, format)
        metadata.bits_per_sample = _get_bits_per_sample(metadata.encoding, metadata.bits_per_sample)
        metadata.encoding = _map_encoding(metadata.encoding)
        return metadata

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
            return load_audio_fileobj(
                uri,
                frame_offset,
                num_frames,
                normalize,
                channels_first,
                format,
                buffer_size,
            )
        else:
            return load_audio(uri, frame_offset, num_frames, normalize, channels_first, format)

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
    ) -> None:
        save_audio(
            uri,
            src,
            sample_rate,
            channels_first,
            format,
            encoding,
            bits_per_sample,
            buffer_size,
        )

    @staticmethod
    def can_decode(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str]) -> bool:
        return True

    @staticmethod
    def can_encode(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str]) -> bool:
        return True


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
    ) -> None:
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
                None,
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


class SoundfileBackend(Backend):
    @abstractmethod
    def info(uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], buffer_size: int = 4096) -> AudioMetaData:
        return soundfile_backend.info(uri, format)

    @abstractmethod
    def load(
        uri: Union[BinaryIO, str, os.PathLike],
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ) -> Tuple[torch.Tensor, int]:
        return soundfile_backend.load(uri, frame_offset, num_frames, normalize, channels_first, format)

    @abstractmethod
    def save(
        uri: Union[BinaryIO, str, os.PathLike],
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
    ) -> None:
        soundfile_backend.save(
            uri, src, sample_rate, channels_first, format=format, encoding=encoding, bits_per_sample=bits_per_sample
        )

    @abstractmethod
    def can_decode(uri, format) -> bool:
        return True

    @abstractmethod
    def can_encode(uri, format) -> bool:
        return True


@lru_cache(None)
def get_available_backends() -> Dict[str, Backend]:
    backend_specs = {}
    if _FFMPEG_INITIALIZED:
        backend_specs["ffmpeg"] = FFmpegBackend
    if _SOX_INITIALIZED:
        backend_specs["sox"] = SoXBackend
    if soundfile_backend._IS_SOUNDFILE_AVAILABLE:
        backend_specs["soundfile"] = SoundfileBackend
    return backend_specs


def get_backend(backend_name, backends) -> Backend:
    if backend := backends.get(backend_name):
        return backend
    else:
        raise ValueError(
            f"Unsupported backend '{backend_name}' specified; ",
            f"please select one of {list(backends.keys())} instead.",
        )


def get_info_func():
    backends = get_available_backends()

    def dispatcher(
        uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], backend_name: Optional[str]
    ) -> Backend:
        if backend_name is not None:
            return get_backend(backend_name, backends)

        for backend in backends.values():
            if backend.can_decode(uri, format):
                return backend
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")

    def info(
        uri: Union[BinaryIO, str, os.PathLike],
        format: Optional[str] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
    ) -> AudioMetaData:
        """Get signal information of an audio file.

        Args:
            uri (path-like object or file-like object):
                Source of audio data. The following types are accepted:

                    * ``path-like``: file path
                    * ``file-like``: Object with ``read(size: int) -> bytes`` method,
                      which returns byte string of at most ``size`` length.

                Note:
                    When the input type is file-like object, this function cannot
                    get the correct length (``num_samples``) for certain formats,
                    such as ``vorbis``.
                    In this case, the value of ``num_samples`` is ``0``.

            format (str or None, optional):
                If not ``None``, interpreted as hint that may allow backend to override the detected format.
                (Default: ``None``)

            buffer_size (int, optional):
                Size of buffer to use when processing file-like objects, in bytes. (Default: ``4096``)

            backend (str or None, optional):
                I/O backend to use. If ``None``, function selects backend given input and available backends.
                Otherwise, must be one of ["ffmpeg", "sox", "soundfile"], with the corresponding backend available.
                (Default: ``None``)

        Returns:
            AudioMetaData: Metadata of the given audio.
        """
        backend = dispatcher(uri, format, backend)
        return backend.info(uri, format, buffer_size)

    return info


def get_load_func():
    backends = get_available_backends()

    def dispatcher(
        uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], backend_name: Optional[str]
    ) -> Backend:
        if backend_name is not None:
            return get_backend(backend_name, backends)

        for backend in backends.values():
            if backend.can_decode(uri, format):
                return backend
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")

    def load(
        uri: Union[BinaryIO, str, os.PathLike],
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: Optional[str] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Load audio data from file.

        Note:
            The formats this function can handle depend on backend availability.
            This function is tested on the following formats:

            * WAV

                * 32-bit floating-point
                * 32-bit signed integer
                * 24-bit signed integer
                * 16-bit signed integer
                * 8-bit unsigned integer

            * FLAC
            * OGG/VORBIS
            * SPHERE

        By default (``normalize=True``, ``channels_first=True``), this function returns Tensor with
        ``float32`` dtype, and the shape of `[channel, time]`.

        .. warning::

            ``normalize`` argument does not perform volume normalization.
            It only converts the sample type to `torch.float32` from the native sample
            type.

            When the input format is WAV with integer type, such as 32-bit signed integer, 16-bit
            signed integer, 24-bit signed integer, and 8-bit unsigned integer, by providing ``normalize=False``,
            this function can return integer Tensor, where the samples are expressed within the whole range
            of the corresponding dtype, that is, ``int32`` tensor for 32-bit signed PCM,
            ``int16`` for 16-bit signed PCM and ``uint8`` for 8-bit unsigned PCM. Since torch does not
            support ``int24`` dtype, 24-bit signed PCM are converted to ``int32`` tensors.

            ``normalize`` argument has no effect on 32-bit floating-point WAV and other formats, such as
            ``flac`` and ``mp3``.

            For these formats, this function always returns ``float32`` Tensor with values.


        Args:
            uri (path-like object or file-like object):
                Source of audio data.
            frame_offset (int, optional):
                Number of frames to skip before start reading data.
            num_frames (int, optional):
                Maximum number of frames to read. ``-1`` reads all the remaining samples,
                starting from ``frame_offset``.
                This function may return the less number of frames if there is not enough
                frames in the given file.
            normalize (bool, optional):
                When ``True``, this function converts the native sample type to ``float32``.
                Default: ``True``.

                If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
                integer type.
                This argument has no effect for formats other than integer WAV type.

            channels_first (bool, optional):
                When True, the returned Tensor has dimension `[channel, time]`.
                Otherwise, the returned Tensor's dimension is `[time, channel]`.

            format (str or None, optional):
                If not ``None``, interpreted as hint that may allow backend to override the detected format.
                (Default: ``None``)

            buffer_size (int, optional):
                Size of buffer to use when processing file-like objects, in bytes. (Default: ``4096``)

            backend (str or None, optional):
                I/O backend to use. If ``None``, function selects backend given input and available backends.
                Otherwise, must be one of ["ffmpeg", "sox", "soundfile"], with the corresponding
                backend being available. (Default: ``None``)

        Returns:
            (torch.Tensor, int): Resulting Tensor and sample rate.
                If the input file has integer wav format and normalization is off, then it has
                integer type, else ``float32`` type. If ``channels_first=True``, it has
                `[channel, time]` else `[time, channel]`.
        """
        backend = dispatcher(uri, format, backend)
        return backend.load(uri, frame_offset, num_frames, normalize, channels_first, format, buffer_size)

    return load


def get_save_func():
    backends = get_available_backends()

    def dispatcher(
        uri: Union[BinaryIO, str, os.PathLike], format: Optional[str], backend_name: Optional[str]
    ) -> Backend:
        if backend_name is not None:
            return get_backend(backend_name, backends)

        for backend in backends.values():
            if backend.can_encode(uri, format):
                return backend
        raise RuntimeError(f"Couldn't find appropriate backend to handle uri {uri} and format {format}.")

    def save(
        uri: Union[BinaryIO, str, os.PathLike],
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
    ):
        """Save audio data to file.

        Note:
            The formats this function can handle depend on the availability of backends.
            This function is tested on the following formats:

            * WAV

                * 32-bit floating-point
                * 32-bit signed integer
                * 16-bit signed integer
                * 8-bit unsigned integer

            * FLAC
            * OGG/VORBIS

        Args:
            uri (str or pathlib.Path): Path to audio file.
            src (torch.Tensor): Audio data to save. must be 2D tensor.
            sample_rate (int): sampling rate
            channels_first (bool, optional): If ``True``, the given tensor is interpreted as `[channel, time]`,
                otherwise `[time, channel]`.
            format (str or None, optional): Override the audio format.
                When ``uri`` argument is path-like object, audio format is
                inferred from file extension. If the file extension is missing or
                different, you can specify the correct format with this argument.

                When ``uri`` argument is file-like object,
                this argument is required.

                Valid values are ``"wav"``, ``"ogg"``, and ``"flac"``.
            encoding (str or None, optional): Changes the encoding for supported formats.
                This argument is effective only for supported formats, i.e.
                ``"wav"`` and ``""flac"```. Valid values are

                    - ``"PCM_S"`` (signed integer Linear PCM)
                    - ``"PCM_U"`` (unsigned integer Linear PCM)
                    - ``"PCM_F"`` (floating point PCM)
                    - ``"ULAW"`` (mu-law)
                    - ``"ALAW"`` (a-law)

            bits_per_sample (int or None, optional): Changes the bit depth for the
                supported formats.
                When ``format`` is one of ``"wav"`` and ``"flac"``,
                you can change the bit depth.
                Valid values are ``8``, ``16``, ``24``, ``32`` and ``64``.

            buffer_size (int, optional):
                Size of buffer to use when processing file-like objects, in bytes. (Default: ``4096``)

            backend (str or None, optional):
                I/O backend to use. If ``None``, function selects backend given input and available backends.
                Otherwise, must be one of ["ffmpeg", "sox", "soundfile"], with the corresponding
                backend being available. (Default: ``None``)



        Supported formats/encodings/bit depth/compression are:

        ``"wav"``
            - 32-bit floating-point PCM
            - 32-bit signed integer PCM
            - 24-bit signed integer PCM
            - 16-bit signed integer PCM
            - 8-bit unsigned integer PCM
            - 8-bit mu-law
            - 8-bit a-law

            Note:
                Default encoding/bit depth is determined by the dtype of
                the input Tensor.

        ``"flac"``
            - 16-bit (default)
            - 24-bit

        ``"ogg"``
            - Doesn't accept changing configuration.
        """
        backend = dispatcher(uri, format, backend)
        return backend.save(uri, src, sample_rate, channels_first, format, encoding, bits_per_sample, buffer_size)

    return save
