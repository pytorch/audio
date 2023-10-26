from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union

import torch
import torio

ffmpeg_ext = torio._extension.lazy_import_ffmpeg_ext()


@dataclass
class CodecConfig:
    """Codec configuration."""

    bit_rate: int = -1
    """Bit rate"""

    compression_level: int = -1
    """Compression level"""

    qscale: Optional[int] = None
    """Global quality factor. Enables variable bit rate. Valid values depend on encoder.

    For example: MP3 takes ``0`` - ``9`` (https://trac.ffmpeg.org/wiki/Encode/MP3) while
    libvorbis takes ``-1`` - ``10``.
    """

    gop_size: int = -1
    """The number of pictures in a group of pictures, or 0 for intra_only"""

    max_b_frames: int = -1
    """maximum number of B-frames between non-B-frames."""


def _convert_config(cfg: CodecConfig):
    if cfg is None:
        return None
    # Convert the codecconfig to C++ compatible type.
    # omitting the return type annotation so as not to access ffmpeg_ext here.
    return ffmpeg_ext.CodecConfig(
        cfg.bit_rate,
        cfg.compression_level,
        cfg.qscale,
        cfg.gop_size,
        cfg.max_b_frames,
    )


def _format_doc(**kwargs):
    def decorator(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj

    return decorator


_encoder = """The name of the encoder to be used.
                When provided, use the specified encoder instead of the default one.

                To list the available encoders, please use
                :py:func:`~torio.utils.ffmpeg_utils.get_audio_encoders` for audio, and
                :py:func:`~torio.utils.ffmpeg_utils.get_video_encoders` for video.

                Default: ``None``."""


_encoder_option = """Options passed to encoder.
                Mapping from str to str.

                To list encoder options for a encoder, you can use
                ``ffmpeg -h encoder=<ENCODER>`` command.

                Default: ``None``.

                |

                In addition to encoder-specific options, you can also pass options related
                to multithreading. They are effective only if the encoder support them.
                If neither of them are provided, StreamReader defaults to single thread.

                ``"threads"``: The number of threads (in str).
                Providing the value ``"0"`` will let FFmpeg decides based on its heuristics.

                ``"thread_type"``: Which multithreading method to use.
                The valid values are ``"frame"`` or ``"slice"``.
                Note that each encoder supports different set of methods.
                If not provided, a default value is used.

                - ``"frame"``: Encode more than one frame at once.
                  Each thread handles one frame.
                  This will increase decoding delay by one frame per thread
                - ``"slice"``: Encode more than one part of a single frame at once.

                |
                """


_encoder_format = """Format used to encode media.
                When encoder supports multiple formats, passing this argument will override
                the format used for encoding.

                To list supported formats for the encoder, you can use
                ``ffmpeg -h encoder=<ENCODER>`` command.

                Default: ``None``.

                Note:
                    When ``encoder_format`` option is not provided, encoder uses its default format.

                    For example, when encoding audio into wav format, 16-bit signed integer is used,
                    and when encoding video into mp4 format (h264 encoder), one of YUV format is used.

                    This is because typically, 32-bit or 16-bit floating point is used in audio models but
                    they are not commonly used in audio formats. Similarly, RGB24 is commonly used in vision
                    models, but video formats usually (and better) support YUV formats.
                """

_codec_config = """Codec configuration. Please refer to :py:class:`CodecConfig` for
                configuration options.

                Default: ``None``."""


_filter_desc = """Additional processing to apply before encoding the input media.
                """

_format_common_args = _format_doc(
    encoder=_encoder,
    encoder_option=_encoder_option,
    encoder_format=_encoder_format,
    codec_config=_codec_config,
    filter_desc=_filter_desc,
)


class StreamingMediaEncoder:
    """Encode and write audio/video streams chunk by chunk

    Args:
        dst (str, path-like or file-like object): The destination where the encoded data are written.
            If string-type, it must be a resource indicator that FFmpeg can
            handle. The supported value depends on the FFmpeg found in the system.

            If file-like object, it must support `write` method with the signature
            `write(data: bytes) -> int`.

            Please refer to the following for the expected signature and behavior of
            `write` method.

            - https://docs.python.org/3/library/io.html#io.BufferedIOBase.write

        format (str or None, optional):
            Override the output format, or specify the output media device.
            Default: ``None`` (no override nor device output).

            This argument serves two different use cases.

            1) Override the output format.
               This is useful when writing raw data or in a format different from the extension.

            2) Specify the output device.
               This allows to output media streams to hardware devices,
               such as speaker and video screen.

            .. note::

               This option roughly corresponds to ``-f`` option of ``ffmpeg`` command.
               Please refer to the ffmpeg documentations for possible values.

               https://ffmpeg.org/ffmpeg-formats.html#Muxers

               Please use :py:func:`~torio.utils.ffmpeg_utils.get_muxers` to list the
               multiplexers available in the current environment.

               For device access, the available values vary based on hardware (AV device) and
               software configuration (ffmpeg build).
               Please refer to the ffmpeg documentations for possible values.

               https://ffmpeg.org/ffmpeg-devices.html#Output-Devices

               Please use :py:func:`~torio.utils.ffmpeg_utils.get_output_devices` to list
               the output devices available in the current environment.

        buffer_size (int):
            The internal buffer size in byte. Used only when `dst` is a file-like object.

            Default: `4096`.
    """

    def __init__(
        self,
        dst: Union[str, Path, BinaryIO],
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ):
        if hasattr(dst, "write"):
            self._s = ffmpeg_ext.StreamingMediaEncoderFileObj(dst, format, buffer_size)
        else:
            self._s = ffmpeg_ext.StreamingMediaEncoder(str(dst), format)
        self._is_open = False

    @_format_common_args
    def add_audio_stream(
        self,
        sample_rate: int,
        num_channels: int,
        format: str = "flt",
        *,
        encoder: Optional[str] = None,
        encoder_option: Optional[Dict[str, str]] = None,
        encoder_sample_rate: Optional[int] = None,
        encoder_num_channels: Optional[int] = None,
        encoder_format: Optional[str] = None,
        codec_config: Optional[CodecConfig] = None,
        filter_desc: Optional[str] = None,
    ):
        """Add an output audio stream.

        Args:
            sample_rate (int): The sample rate.

            num_channels (int): The number of channels.

            format (str, optional): Input sample format, which determines the dtype
                of the input tensor.

                - ``"u8"``: The input tensor must be ``torch.uint8`` type.
                - ``"s16"``: The input tensor must be ``torch.int16`` type.
                - ``"s32"``: The input tensor must be ``torch.int32`` type.
                - ``"s64"``: The input tensor must be ``torch.int64`` type.
                - ``"flt"``: The input tensor must be ``torch.float32`` type.
                - ``"dbl"``: The input tensor must be ``torch.float64`` type.

                Default: ``"flt"``.

            encoder (str or None, optional): {encoder}

            encoder_option (dict or None, optional): {encoder_option}

            encoder_sample_rate (int or None, optional): Override the sample rate used for encoding time.
                Some encoders pose restriction on the sample rate used for encoding.
                If the source sample rate is not supported by the encoder, the source sample rate is used,
                otherwise a default one is picked.

                For example, ``"opus"`` encoder only supports 48k Hz, so, when encoding a
                waveform with ``"opus"`` encoder, it is always encoded as 48k Hz.
                Meanwhile ``"mp3"`` (``"libmp3lame"``) supports 44.1k, 48k, 32k, 22.05k,
                24k, 16k, 11.025k, 12k and 8k Hz.
                If the original sample rate is one of these, then the original sample rate
                is used, otherwise it will be resampled to a default one (44.1k).
                When encoding into WAV format, there is no restriction on sample rate,
                so the original sample rate will be used.

                Providing ``encoder_sample_rate`` will override this behavior and
                make encoder attempt to use the provided sample rate.
                The provided value must be one support by the encoder.

            encoder_num_channels (int or None, optional): Override the number of channels used for encoding.

                Similar to sample rate, some encoders (such as ``"opus"``,
                ``"vorbis"`` and ``"g722"``) pose restriction on
                the numbe of channels that can be used for encoding.

                If the original number of channels is supported by encoder,
                then it will be used, otherwise, the encoder attempts to
                remix the channel to one of the supported ones.

                Providing ``encoder_num_channels`` will override this behavior and
                make encoder attempt to use the provided number of channels.
                The provided value must be one support by the encoder.

            encoder_format (str or None, optional): {encoder_format}

            codec_config (CodecConfig or None, optional): {codec_config}

            filter_desc (str or None, optional): {filter_desc}
        """
        self._s.add_audio_stream(
            sample_rate,
            num_channels,
            format,
            encoder,
            encoder_option,
            encoder_format,
            encoder_sample_rate,
            encoder_num_channels,
            _convert_config(codec_config),
            filter_desc,
        )

    @_format_common_args
    def add_video_stream(
        self,
        frame_rate: float,
        width: int,
        height: int,
        format: str = "rgb24",
        *,
        encoder: Optional[str] = None,
        encoder_option: Optional[Dict[str, str]] = None,
        encoder_frame_rate: Optional[float] = None,
        encoder_width: Optional[int] = None,
        encoder_height: Optional[int] = None,
        encoder_format: Optional[str] = None,
        codec_config: Optional[CodecConfig] = None,
        filter_desc: Optional[str] = None,
        hw_accel: Optional[str] = None,
    ):
        """Add an output video stream.

        This method has to be called before `open` is called.

        Args:
            frame_rate (float): Frame rate of the video.

            width (int): Width of the video frame.

            height (int): Height of the video frame.

            format (str, optional): Input pixel format, which determines the
                color channel order of the input tensor.

                - ``"gray8"``: One channel, grayscale.
                - ``"rgb24"``: Three channels in the order of RGB.
                - ``"bgr24"``: Three channels in the order of BGR.
                - ``"yuv444p"``: Three channels in the order of YUV.

                Default: ``"rgb24"``.

                In either case, the input tensor has to be ``torch.uint8`` type and
                the shape must be (frame, channel, height, width).

            encoder (str or None, optional): {encoder}

            encoder_option (dict or None, optional): {encoder_option}

            encoder_frame_rate (float or None, optional): Override the frame rate used for encoding.

                Some encoders, (such as ``"mpeg1"`` and ``"mpeg2"``) pose restriction on the
                frame rate that can be used for encoding.
                If such case, if the source frame rate (provided as ``frame_rate``) is not
                one of the supported frame rate, then a default one is picked, and the frame rate
                is changed on-the-fly. Otherwise the source frame rate is used.

                Providing ``encoder_frame_rate`` will override this behavior and
                make encoder attempts to use the provided sample rate.
                The provided value must be one support by the encoder.

            encoder_width (int or None, optional): Width of the image used for encoding.
                This allows to change the image size during encoding.

            encoder_height (int or None, optional): Height of the image used for encoding.
                This allows to change the image size during encoding.

            encoder_format (str or None, optional): {encoder_format}

            codec_config (CodecConfig or None, optional): {codec_config}

            filter_desc (str or None, optional): {filter_desc}

            hw_accel (str or None, optional): Enable hardware acceleration.

                When video is encoded on CUDA hardware, for example
                `encoder="h264_nvenc"`, passing CUDA device indicator to `hw_accel`
                (i.e. `hw_accel="cuda:0"`) will make StreamingMediaEncoder expect video
                chunk to be CUDA Tensor. Passing CPU Tensor will result in an error.

                If `None`, the video chunk Tensor has to be CPU Tensor.
                Default: ``None``.
        """
        self._s.add_video_stream(
            frame_rate,
            width,
            height,
            format,
            encoder,
            encoder_option,
            encoder_format,
            encoder_frame_rate,
            encoder_width,
            encoder_height,
            hw_accel,
            _convert_config(codec_config),
            filter_desc,
        )

    def set_metadata(self, metadata: Dict[str, str]):
        """Set file-level metadata

        Args:
            metadata (dict or None, optional): File-level metadata.
        """
        self._s.set_metadata(metadata)

    def _print_output_stream(self, i: int):
        """[debug] Print the registered stream information to stdout."""
        self._s.dump_format(i)

    def open(self, option: Optional[Dict[str, str]] = None) -> "StreamingMediaEncoder":
        """Open the output file / device and write the header.

        :py:class:`StreamingMediaEncoder` is also a context manager and therefore supports the
        ``with`` statement.
        This method returns the instance on which the method is called (i.e. `self`),
        so that it can be used in `with` statement.
        It is recommended to use context manager, as the file is closed automatically
        when exiting from ``with`` clause.

        Args:
            option (dict or None, optional): Private options for protocol, device and muxer. See example.

        Example - Protocol option
            >>> s = StreamingMediaEncoder(dst="rtmp://localhost:1234/live/app", format="flv")
            >>> s.add_video_stream(...)
            >>> # Passing protocol option `listen=1` makes StreamingMediaEncoder act as RTMP server.
            >>> with s.open(option={"listen": "1"}) as f:
            >>>     f.write_video_chunk(...)

        Example - Device option
            >>> s = StreamingMediaEncoder("-", format="sdl")
            >>> s.add_video_stream(..., encoder_format="rgb24")
            >>> # Open SDL video player with fullscreen
            >>> with s.open(option={"window_fullscreen": "1"}):
            >>>     f.write_video_chunk(...)

        Example - Muxer option
            >>> s = StreamingMediaEncoder("foo.flac")
            >>> s.add_audio_stream(...)
            >>> s.set_metadata({"artist": "torio contributors"})
            >>> # FLAC muxer has a private option to not write the header.
            >>> # The resulting file does not contain the above metadata.
            >>> with s.open(option={"write_header": "false"}) as f:
            >>>     f.write_audio_chunk(...)
        """
        if not self._is_open:
            self._s.open(option)
            self._is_open = True
        return self

    def close(self):
        """Close the output

        :py:class:`StreamingMediaEncoder` is also a context manager and therefore supports the
        ``with`` statement.
        It is recommended to use context manager, as the file is closed automatically
        when exiting from ``with`` clause.

        See :py:meth:`StreamingMediaEncoder.open` for more detail.
        """
        if self._is_open:
            self._s.close()
            self._is_open = False

    def write_audio_chunk(self, i: int, chunk: torch.Tensor, pts: Optional[float] = None):
        """Write audio data

        Args:
            i (int): Stream index.
            chunk (Tensor): Waveform tensor. Shape: `(frame, channel)`.
                The ``dtype`` must match what was passed to :py:meth:`add_audio_stream` method.
            pts (float, optional, or None): If provided, overwrite the presentation timestamp.

                .. note::

                   The provided value is converted to integer value expressed in basis of
                   sample rate. Therefore, it is truncated to the nearest value of
                   ``n / sample_rate``.
        """
        self._s.write_audio_chunk(i, chunk, pts)

    def write_video_chunk(self, i: int, chunk: torch.Tensor, pts: Optional[float] = None):
        """Write video/image data

        Args:
            i (int): Stream index.
            chunk (Tensor): Video/image tensor.
                Shape: `(time, channel, height, width)`.
                The ``dtype`` must be ``torch.uint8``.
                The shape (height, width and the number of channels) must match
                what was configured when calling :py:meth:`add_video_stream`
            pts (float, optional or None): If provided, overwrite the presentation timestamp.

                .. note::

                   The provided value is converted to integer value expressed in basis of
                   frame rate. Therefore, it is truncated to the nearest value of
                   ``n / frame_rate``.
        """
        self._s.write_video_chunk(i, chunk, pts)

    def flush(self):
        """Flush the frames from encoders and write the frames to the destination."""
        self._s.flush()

    def __enter__(self):
        """Context manager so that the destination is closed and data are flushed automatically."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Context manager so that the destination is closed and data are flushed automatically."""
        self.flush()
        self.close()
