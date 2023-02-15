from typing import BinaryIO, Dict, Optional, Union

import torch
import torchaudio


def _format_doc(**kwargs):
    def decorator(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj

    return decorator


_encoder = """The name of the encoder to be used.
                When provided, use the specified encoder instead of the default one.

                To list the available encoders, please use
                :py:func:`~torchaudio.utils.ffmpeg_utils.get_audio_encoders` for audio, and
                :py:func:`~torchaudio.utils.ffmpeg_utils.get_video_encoders` for video.

                Default: ``None``."""


_encoder_option = """Options passed to encoder.
                Mapping from str to str.

                To list encoder options for a encoder, you can use
                ``ffmpeg -h encoder=<ENCODER>`` command.

                Default: ``None``."""


_encoder_format = """Format used to encode media.
                When encoder supports multiple formats, passing this argument will override
                the format used for encoding.

                To list supported formats for the encoder, you can use
                ``ffmpeg -h encoder=<ENCODER>`` command.

                Default: ``None``."""


_format_common_args = _format_doc(
    encoder=_encoder,
    encoder_option=_encoder_option,
    encoder_format=_encoder_format,
)


@torchaudio._extension.fail_if_no_ffmpeg
class StreamWriter:
    """Encode and write audio/video streams chunk by chunk

    Args:
        dst (str or file-like object): The destination where the encoded data are written.
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

               Please use :py:func:`~torchaudio.utils.ffmpeg_utils.get_muxers` to list the
               multiplexers available in the current environment.

               For device access, the available values vary based on hardware (AV device) and
               software configuration (ffmpeg build).
               Please refer to the ffmpeg documentations for possible values.

               https://ffmpeg.org/ffmpeg-devices.html#Output-Devices

               Please use :py:func:`~torchaudio.utils.ffmpeg_utils.get_output_devices` to list
               the output devices available in the current environment.

        buffer_size (int):
            The internal buffer size in byte. Used only when `dst` is a file-like object.

            Default: `4096`.
    """

    def __init__(
        self,
        dst: Union[str, BinaryIO],
        format: Optional[str] = None,
        buffer_size: int = 4096,
    ):
        torch._C._log_api_usage_once("torchaudio.io.StreamWriter")
        if isinstance(dst, str):
            self._s = torch.classes.torchaudio.ffmpeg_StreamWriter(dst, format)
        elif hasattr(dst, "write"):
            self._s = torchaudio.lib._torchaudio_ffmpeg.StreamWriterFileObj(dst, format, buffer_size)
        else:
            raise ValueError("`dst` must be either a string or a file-like object.")
        self._is_open = False

    @_format_common_args
    def add_audio_stream(
        self,
        sample_rate: int,
        num_channels: int,
        format: str = "flt",
        encoder: Optional[str] = None,
        encoder_option: Optional[Dict[str, str]] = None,
        encoder_format: Optional[str] = None,
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

            encoder_format (str or None, optional): {encoder_format}
        """
        self._s.add_audio_stream(sample_rate, num_channels, format, encoder, encoder_option, encoder_format)

    @_format_common_args
    def add_video_stream(
        self,
        frame_rate: float,
        width: int,
        height: int,
        format: str = "rgb24",
        encoder: Optional[str] = None,
        encoder_option: Optional[Dict[str, str]] = None,
        encoder_format: Optional[str] = None,
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

            encoder_format (str or None, optional): {encoder_format}

            hw_accel (str or None, optional): Enable hardware acceleration.

                When video is encoded on CUDA hardware, for example
                `encoder="h264_nvenc"`, passing CUDA device indicator to `hw_accel`
                (i.e. `hw_accel="cuda:0"`) will make StreamWriter expect video
                chunk to be CUDA Tensor. Passing CPU Tensor will result in an error.

                If `None`, the video chunk Tensor has to be CPU Tensor.
                Default: ``None``.
        """
        self._s.add_video_stream(frame_rate, width, height, format, encoder, encoder_option, encoder_format, hw_accel)

    def set_metadata(self, metadata: Dict[str, str]):
        """Set file-level metadata

        Args:
            metadata (dict or None, optional): File-level metadata.
        """
        self._s.set_metadata(metadata)

    def _print_output_stream(self, i: int):
        """[debug] Print the registered stream information to stdout."""
        self._s.dump_format(i)

    def open(self, option: Optional[Dict[str, str]] = None) -> "StreamWriter":
        """Open the output file / device and write the header.

        :py:class:`StreamWriter` is also a context manager and therefore supports the
        ``with`` statement.
        This method returns the instance on which the method is called (i.e. `self`),
        so that it can be used in `with` statement.
        It is recommended to use context manager, as the file is closed automatically
        when exiting from ``with`` clause.

        Args:
            option (dict or None, optional): Private options for protocol, device and muxer. See example.

        Example - Protocol option
            >>> s = StreamWriter(dst="rtmp://localhost:1234/live/app", format="flv")
            >>> s.add_video_stream(...)
            >>> # Passing protocol option `listen=1` makes StreamWriter act as RTMP server.
            >>> with s.open(option={"listen": "1"}) as f:
            >>>     f.write_video_chunk(...)

        Example - Device option
            >>> s = StreamWriter("-", format="sdl")
            >>> s.add_video_stream(..., encoder_format="rgb24")
            >>> # Open SDL video player with fullscreen
            >>> with s.open(option={"window_fullscreen": "1"}):
            >>>     f.write_video_chunk(...)

        Example - Muxer option
            >>> s = StreamWriter("foo.flac")
            >>> s.add_audio_stream(...)
            >>> s.set_metadata({"artist": "torchaudio contributors"})
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

        :py:class:`StreamWriter` is also a context manager and therefore supports the
        ``with`` statement.
        It is recommended to use context manager, as the file is closed automatically
        when exiting from ``with`` clause.

        See :py:meth:`StreamWriter.open` for more detail.
        """
        if self._is_open:
            self._s.close()
            self._is_open = False

    def write_audio_chunk(self, i: int, chunk: torch.Tensor):
        """Write audio data

        Args:
            i (int): Stream index.
            chunk (Tensor): Waveform tensor. Shape: `(frame, channel)`.
                The ``dtype`` must match what was passed to :py:meth:`add_audio_stream` method.
        """
        self._s.write_audio_chunk(i, chunk)

    def write_video_chunk(self, i: int, chunk: torch.Tensor):
        """Write video/image data

        Args:
            i (int): Stream index.
            chunk (Tensor): Video/image tensor.
                Shape: `(time, channel, height, width)`.
                The ``dtype`` must be ``torch.uint8``.
                The shape (height, width and the number of channels) must match
                what was configured when calling :py:meth:`add_video_stream`
        """
        self._s.write_video_chunk(i, chunk)

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
