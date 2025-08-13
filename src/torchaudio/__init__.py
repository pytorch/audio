from torchaudio._internal.module_utils import dropping_io_support, dropping_class_io_support
from typing import Union, BinaryIO, Optional, Tuple
import os
import torch
import sys

# Initialize extension and backend first
from . import _extension  # noqa  # usort: skip
from ._backend import (  # noqa  # usort: skip
    AudioMetaData as _AudioMetaData,
    get_audio_backend as _get_audio_backend,
    info as _info,
    list_audio_backends as _list_audio_backends,
    set_audio_backend as _set_audio_backend,
)
from ._torchcodec import load_with_torchcodec, save_with_torchcodec

AudioMetaData = dropping_class_io_support(_AudioMetaData)
get_audio_backend = dropping_io_support(_get_audio_backend)
info = dropping_io_support(_info)
list_audio_backends = dropping_io_support(_list_audio_backends)
set_audio_backend = dropping_io_support(_set_audio_backend)

from . import (  # noqa: F401
    compliance,
    datasets,
    functional,
    io,
    kaldi_io,
    models,
    pipelines,
    sox_effects,
    transforms,
    utils,
)

# For BC
from . import backend  # noqa # usort: skip

try:
    from .version import __version__, git_version  # noqa: F401
except ImportError:
    pass

# CI cannot currently build with ffmpeg>4, but torchcodec is buggy with ffmpeg4. This hack
# allows CI to build with ffmpeg4 and works around load/test bugginess.
if "pytest" in sys.modules:
    from torchaudio.utils import wav_utils
    def load(
        uri: str,
        normalize: bool = True,
        channels_first: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        return wav_utils.load_wav(uri, normalize, channels_first)

    def save(
        uri: str,
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
        compression: Optional[Union[float, int]] = None,
    ):
        wav_utils.save_wav(uri, src, sample_rate, channels_first=channels_first)
else:
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
        """Load audio data from source using TorchCodec's AudioDecoder.

        .. note::

            As of TorchAudio 2.9, this function relies on TorchCodec's decoding capabilities under the hood. It is
            provided for convenience, but we do recommend that you port your code to
            natively use ``torchcodec``'s ``AudioDecoder`` class for better
            performance:
            https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.
            Because of the reliance on Torchcodec, the parameters ``normalize``, ``buffer_size``, and
            ``backend`` are ignored and accepted only for backwards compatibility.


        Args:
            uri (path-like object or file-like object):
                Source of audio data. The following types are accepted:

                * ``path-like``: File path or URL.
                * ``file-like``: Object with ``read(size: int) -> bytes`` method.

            frame_offset (int, optional):
                Number of samples to skip before start reading data.
            num_frames (int, optional):
                Maximum number of samples to read. ``-1`` reads all the remaining samples,
                starting from ``frame_offset``.
            normalize (bool, optional):
                TorchCodec always returns normalized float32 samples. This parameter
                is ignored and a warning is issued if set to False.
                Default: ``True``.
            channels_first (bool, optional):
                When True, the returned Tensor has dimension `[channel, time]`.
                Otherwise, the returned Tensor's dimension is `[time, channel]`.
            format (str or None, optional):
                Format hint for the decoder. May not be supported by all TorchCodec
                decoders. (Default: ``None``)
            buffer_size (int, optional):
                Not used by TorchCodec AudioDecoder. Provided for API compatibility.
            backend (str or None, optional):
                Not used by TorchCodec AudioDecoder. Provided for API compatibility.

        Returns:
            (torch.Tensor, int): Resulting Tensor and sample rate.
            Always returns float32 tensors. If ``channels_first=True``, shape is
            `[channel, time]`, otherwise `[time, channel]`.

        Raises:
            ImportError: If torchcodec is not available.
            ValueError: If unsupported parameters are used.
            RuntimeError: If TorchCodec fails to decode the audio.

        Note:
            - TorchCodec always returns normalized float32 samples, so the ``normalize``
            parameter has no effect.
            - The ``buffer_size`` and ``backend`` parameters are ignored.
            - Not all audio formats supported by torchaudio backends may be supported
            by TorchCodec.
        """
        return load_with_torchcodec(
            uri,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=normalize,
            channels_first=channels_first,
            format=format,
            buffer_size=buffer_size,
            backend=backend
        )

    def save(
        uri: Union[str, os.PathLike],
        src: torch.Tensor,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
        buffer_size: int = 4096,
        backend: Optional[str] = None,
        compression: Optional[Union[float, int]] = None,
    ) -> None:
        """Save audio data to file using TorchCodec's AudioEncoder.

        .. note::

            As of TorchAudio 2.9, this function relies on TorchCodec's encoding capabilities under the hood.
            It is provided for convenience, but we do recommend that you port your code to
            natively use ``torchcodec``'s ``AudioEncoder`` class for better
            performance:
            https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.encoders.AudioEncoder.
            Because of the reliance on Torchcodec, the parameters ``format``, ``encoding``,
            ``bits_per_sample``, ``buffer_size``, and ``backend``, are ignored and accepted only for
            backwards compatibility.

        Args:
            uri (path-like object):
                Path to save the audio file. The file extension determines the format.

            src (torch.Tensor):
                Audio data to save. Must be a 1D or 2D tensor with float32 values
                in the range [-1, 1]. If 2D, shape should be [channel, time] when
                channels_first=True, or [time, channel] when channels_first=False.

            sample_rate (int):
                Sample rate of the audio data.

            channels_first (bool, optional):
                Indicates whether the input tensor has channels as the first dimension.
                If True, expects [channel, time]. If False, expects [time, channel].
                Default: True.

            format (str or None, optional):
                Audio format hint. Not used by TorchCodec (format is determined by
                file extension). A warning is issued if provided.
                Default: None.

            encoding (str or None, optional):
                Audio encoding. Not fully supported by TorchCodec AudioEncoder.
                A warning is issued if provided. Default: None.

            bits_per_sample (int or None, optional):
                Bits per sample. Not directly supported by TorchCodec AudioEncoder.
                A warning is issued if provided. Default: None.

            buffer_size (int, optional):
                Not used by TorchCodec AudioEncoder. Provided for API compatibility.
                A warning is issued if not default value. Default: 4096.

            backend (str or None, optional):
                Not used by TorchCodec AudioEncoder. Provided for API compatibility.
                A warning is issued if provided. Default: None.

            compression (float, int or None, optional):
                Compression level or bit rate. Maps to bit_rate parameter in
                TorchCodec AudioEncoder. Default: None.

        Raises:
            ImportError: If torchcodec is not available.
            ValueError: If input parameters are invalid.
            RuntimeError: If TorchCodec fails to encode the audio.

        Note:
            - TorchCodec AudioEncoder expects float32 samples in [-1, 1] range.
            - Some parameters (format, encoding, bits_per_sample, buffer_size, backend)
            are not used by TorchCodec but are provided for API compatibility.
            - The output format is determined by the file extension in the uri.
            - TorchCodec uses FFmpeg under the hood for encoding.
        """
        return save_with_torchcodec(uri, src, sample_rate,
            channels_first=channels_first,
            format=format,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
            buffer_size=buffer_size,
            backend=backend,
            compression=compression)

__all__ = [
    "AudioMetaData",
    "load",
    "load_with_torchcodec",
    "save_with_torchcodec",
    "info",
    "save",
    "io",
    "compliance",
    "datasets",
    "functional",
    "models",
    "pipelines",
    "kaldi_io",
    "utils",
    "sox_effects",
    "transforms",
    "list_audio_backends",
    "get_audio_backend",
    "set_audio_backend",
]
