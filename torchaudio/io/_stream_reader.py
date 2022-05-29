from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

import torch
import torchaudio


@dataclass
class StreamReaderSourceStream:
    """StreamReaderSourceStream()

    The metadata of a source stream. This class is used when representing streams of
    media type other than `audio` or `video`.

    When source stream is `audio` or `video` type, :py:class:`SourceAudioStream` and
    :py:class:`SourceVideoStream`, which reports additional media-specific attributes,
    are used respectively.
    """

    media_type: str
    """The type of the stream.
    One of `audio`, `video`, `data`, `subtitle`, `attachment` and empty string.

    .. note::
       Only `audio` and `video` streams are supported for output.
    .. note::
       Still images, such as PNG and JPEG formats are reported as `video`.
    """
    codec: str
    """Short name of the codec. Such as ``"pcm_s16le"`` and ``"h264"``."""
    codec_long_name: str
    """Detailed name of the codec.

    Such as "`PCM signed 16-bit little-endian`" and "`H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10`".
    """
    format: Optional[str]
    """Media format. Such as ``"s16"`` and ``"yuv420p"``.

    Commonly found audio values are;

    - ``"u8"``, ``"u8p"``: Unsigned 8-bit unsigned interger.
    - ``"s16"``, ``"s16p"``: 16-bit signed integer.
    - ``"s32"``, ``"s32p"``: 32-bit signed integer.
    - ``"flt"``, ``"fltp"``: 32-bit floating-point.

    .. note::

       `p` at the end indicates the format is `planar`.
       Channels are grouped together instead of interspersed in memory.
    """
    bit_rate: Optional[int]
    """Bit rate of the stream in bits-per-second.
    This is an estimated values based on the initial few frames of the stream.
    For container formats and variable bit rate, it can be 0.
    """
    num_frames: Optional[int]
    """The number of frames in the stream"""
    bits_per_sample: Optional[int]
    """This is the number of valid bits in each output sample.
    For compressed format, it can be 0.
    """


@dataclass
class StreamReaderSourceAudioStream(StreamReaderSourceStream):
    """StreamReaderSourceAudioStream()

    The metadata of an audio source stream.

    In addition to the attributes reported by :py:func:`StreamReaderSourceStream`,
    when the source stream is audio type, then the following additional attributes
    are reported.
    """

    sample_rate: float
    """Sample rate of the audio."""
    num_channels: int
    """Number of channels."""


@dataclass
class StreamReaderSourceVideoStream(StreamReaderSourceStream):
    """StreamReaderSourceVideoStream()

    The metadata of a video source stream.

    In addition to the attributes reported by :py:func:`StreamReaderSourceStream`,
    when the source stream is audio type, then the following additional attributes
    are reported.
    """

    width: int
    """Width of the video frame in pixel."""
    height: int
    """Height of the video frame in pixel."""
    frame_rate: float
    """Frame rate."""


# Indices of SrcInfo returned by low-level `get_src_stream_info`
# - COMMON
_MEDIA_TYPE = 0
_CODEC = 1
_CODEC_LONG = 2
_FORMAT = 3
_BIT_RATE = 4
_NUM_FRAMES = 5
_BPS = 6
# - AUDIO
_SAMPLE_RATE = 7
_NUM_CHANNELS = 8
# - VIDEO
_WIDTH = 9
_HEIGHT = 10
_FRAME_RATE = 11


def _parse_si(i):
    media_type = i[_MEDIA_TYPE]
    codec_name = i[_CODEC]
    codec_long_name = i[_CODEC_LONG]
    fmt = i[_FORMAT]
    bit_rate = i[_BIT_RATE]
    num_frames = i[_NUM_FRAMES]
    bps = i[_BPS]
    if media_type == "audio":
        return StreamReaderSourceAudioStream(
            media_type=media_type,
            codec=codec_name,
            codec_long_name=codec_long_name,
            format=fmt,
            bit_rate=bit_rate,
            num_frames=num_frames,
            bits_per_sample=bps,
            sample_rate=i[_SAMPLE_RATE],
            num_channels=i[_NUM_CHANNELS],
        )
    if media_type == "video":
        return StreamReaderSourceVideoStream(
            media_type=media_type,
            codec=codec_name,
            codec_long_name=codec_long_name,
            format=fmt,
            bit_rate=bit_rate,
            num_frames=num_frames,
            bits_per_sample=bps,
            width=i[_WIDTH],
            height=i[_HEIGHT],
            frame_rate=i[_FRAME_RATE],
        )
    return StreamReaderSourceStream(
        media_type=media_type,
        codec=codec_name,
        codec_long_name=codec_long_name,
        format=None,
        bit_rate=None,
        num_frames=None,
        bits_per_sample=None,
    )


@dataclass
class StreamReaderOutputStream:
    """OutputStream()

    Output stream configured on :py:class:`StreamReader`.
    """

    source_index: int
    """Index of the source stream that this output stream is connected."""
    filter_description: str
    """Description of filter graph applied to the source stream."""


def _parse_oi(i):
    return StreamReaderOutputStream(i[0], i[1])


def _get_afilter_desc(sample_rate: Optional[int], fmt: Optional[str]):
    descs = []
    if sample_rate is not None:
        descs.append(f"aresample={sample_rate}")
    if fmt is not None:
        descs.append(f"aformat=sample_fmts={fmt}")
    return ",".join(descs) if descs else None


def _get_vfilter_desc(frame_rate: Optional[float], width: Optional[int], height: Optional[int], fmt: Optional[str]):
    descs = []
    if frame_rate is not None:
        descs.append(f"fps={frame_rate}")
    scales = []
    if width is not None:
        scales.append(f"width={width}")
    if height is not None:
        scales.append(f"height={height}")
    if scales:
        descs.append(f"scale={':'.join(scales)}")
    if fmt is not None:
        descs.append(f"format=pix_fmts={fmt}")
    return ",".join(descs) if descs else None


def _format_doc(**kwargs):
    def decorator(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj

    return decorator


_frames_per_chunk = """Number of frames returned as one chunk.
                If the source stream is exhausted before enough frames are buffered,
                then the chunk is returned as-is."""

_buffer_chunk_size = """Internal buffer size.
                When the number of chunks buffered exceeds this number, old frames are
                dropped.

                Default: ``3``."""

_audio_stream_index = """The source audio stream index.
                If omitted, :py:attr:`default_audio_stream` is used."""


_video_stream_index = """The source video stream index.
                If omitted, :py:attr:`default_video_stream` is used."""

_decoder = """The name of the decoder to be used.
                When provided, use the specified decoder instead of the default one.

                To list the available decoders, you can use `ffmpeg -decoders` command.

                Default: ``None``."""

_decoder_option = """Options passed to decoder.
                Mapping from str to str.

                To list decoder options for a decoder, you can use
                `ffmpeg -h decoder=<DECODER>` command.

                Default: ``None``."""


_hw_accel = """Enable hardware acceleration.

                When video is decoded on CUDA hardware, for example
                `decode="h264_cuvid"`, passing CUDA device indicator to `hw_accel`
                (i.e. `hw_accel="cuda:0"`) will place the resulting frames
                directly on the specifiec CUDA device.

                If `None`, the frame will be moved to CPU memory.
                Default: ``None``."""


_format_audio_args = _format_doc(
    frames_per_chunk=_frames_per_chunk,
    buffer_chunk_size=_buffer_chunk_size,
    stream_index=_audio_stream_index,
    decoder=_decoder,
    decoder_option=_decoder_option,
)


_format_video_args = _format_doc(
    frames_per_chunk=_frames_per_chunk,
    buffer_chunk_size=_buffer_chunk_size,
    stream_index=_video_stream_index,
    decoder=_decoder,
    decoder_option=_decoder_option,
    hw_accel=_hw_accel,
)


class StreamReader:
    """Fetch and decode audio/video streams chunk by chunk.

    For the detailed usage of this class, please refer to the tutorial.

    Args:
        src (str or file-like object): The media source.
            If string-type, it must be a resource indicator that FFmpeg can
            handle. This includes a file path, URL, device identifier or
            filter expression. The supported value depends on the FFmpeg found
            in the system.

            If file-like object, it must support `read` method with the signature
            `read(size: int) -> bytes`.
            Additionally, if the file-like object has `seek` method, it uses
            the method when parsing media metadata. This improves the reliability
            of codec detection. The signagure of `seek` method must be
            `seek(offset: int, whence: int) -> int`.

            Please refer to the following for the expected signature and behavior
            of `read` and `seek` method.

            - https://docs.python.org/3/library/io.html#io.BufferedIOBase.read
            - https://docs.python.org/3/library/io.html#io.IOBase.seek

        format (str or None, optional):
            Override the input format, or specify the source sound device.
            Default: ``None`` (no override nor device input).

            This argument serves two different usecases.

            1) Override the source format.
               This is useful when the input data do not contain a header.

            2) Specify the input source device.
               This allows to load media stream from hardware devices,
               such as microphone, camera and screen, or a virtual device.


            .. note::

               This option roughly corresponds to ``-f`` option of ``ffmpeg`` command.
               Please refer to the ffmpeg documentations for the possible values.

               https://ffmpeg.org/ffmpeg-formats.html

               For device access, the available values vary based on hardware (AV device) and
               software configuration (ffmpeg build).

               https://ffmpeg.org/ffmpeg-devices.html

        option (dict of str to str, optional):
            Custom option passed when initializing format context (opening source).

            You can use this argument to change the input source before it is passed to decoder.

            Default: ``None``.

        buffer_size (int):
            The internal buffer size in byte. Used only when `src` is file-like object.

            Default: `4096`.
    """

    def __init__(
        self,
        src: str,
        format: Optional[str] = None,
        option: Optional[Dict[str, str]] = None,
        buffer_size: int = 4096,
    ):
        if isinstance(src, str):
            self._be = torch.classes.torchaudio.ffmpeg_StreamReader(src, format, option)
        elif hasattr(src, "read"):
            self._be = torchaudio._torchaudio_ffmpeg.StreamReaderFileObj(src, format, option, buffer_size)
        else:
            raise ValueError("`src` must be either string or file-like object.")

        i = self._be.find_best_audio_stream()
        self._default_audio_stream = None if i < 0 else i
        i = self._be.find_best_video_stream()
        self._default_video_stream = None if i < 0 else i

    @property
    def num_src_streams(self):
        """Number of streams found in the provided media source.

        :type: int
        """
        return self._be.num_src_streams()

    @property
    def num_out_streams(self):
        """Number of output streams configured by client code.

        :type: int
        """
        return self._be.num_out_streams()

    @property
    def default_audio_stream(self):
        """The index of default audio stream. ``None`` if there is no audio stream

        :type: Optional[int]
        """
        return self._default_audio_stream

    @property
    def default_video_stream(self):
        """The index of default video stream. ``None`` if there is no video stream

        :type: Optional[int]
        """
        return self._default_video_stream

    def get_src_stream_info(self, i: int) -> torchaudio.io.StreamReaderSourceStream:
        """Get the metadata of source stream

        Args:
            i (int): Stream index.
        Returns:
            SourceStream
        """
        return _parse_si(self._be.get_src_stream_info(i))

    def get_out_stream_info(self, i: int) -> torchaudio.io.StreamReaderOutputStream:
        """Get the metadata of output stream

        Args:
            i (int): Stream index.
        Returns:
            OutputStream
        """
        return _parse_oi(self._be.get_out_stream_info(i))

    def seek(self, timestamp: float):
        """Seek the stream to the given timestamp [second]

        Args:
            timestamp (float): Target time in second.
        """
        self._be.seek(timestamp)

    @_format_audio_args
    def add_basic_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        format: Optional[str] = "fltp",
        sample_rate: Optional[int] = None,
    ):
        """Add output audio stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            format (str, optional): Output sample format (precision).

                If ``None``, the output chunk has dtype corresponding to
                the precision of the source audio.

                Otherwise, the sample is converted and the output dtype is changed
                as following.

                - ``"u8p"``: The output is ``torch.uint8`` type.
                - ``"s16p"``: The output is ``torch.int16`` type.
                - ``"s32p"``: The output is ``torch.int32`` type.
                - ``"s64p"``: The output is ``torch.int64`` type.
                - ``"fltp"``: The output is ``torch.float32`` type.
                - ``"dblp"``: The output is ``torch.float64`` type.

                Default: ``"fltp"``.

            sample_rate (int or None, optional): If provided, resample the audio.
        """
        self.add_audio_stream(
            frames_per_chunk,
            buffer_chunk_size,
            stream_index,
            decoder,
            decoder_option,
            _get_afilter_desc(sample_rate, format),
        )

    @_format_video_args
    def add_basic_video_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        hw_accel: Optional[str] = None,
        format: Optional[str] = "rgb24",
        frame_rate: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """Add output video stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            hw_accel (str or None, optional): {hw_accel}

            format (str, optional): Change the format of image channels. Valid values are,

                - ``"rgb24"``: 8 bits * 3 channels (R, G, B)
                - ``"bgr24"``: 8 bits * 3 channels (B, G, R)
                - ``"yuv420p"``: 8 bits * 3 channels (Y, U, V)
                - ``"gray"``: 8 bits * 1 channels

                Default: ``"rgb24"``.

            frame_rate (int or None, optional): If provided, change the frame rate.

            width (int or None, optional): If provided, change the image width. Unit: Pixel.

            height (int or None, optional): If provided, change the image height. Unit: Pixel.
        """
        self.add_video_stream(
            frames_per_chunk,
            buffer_chunk_size,
            stream_index,
            decoder,
            decoder_option,
            hw_accel,
            _get_vfilter_desc(frame_rate, width, height, format),
        )

    @_format_audio_args
    def add_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        filter_desc: Optional[str] = None,
    ):
        """Add output audio stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            filter_desc (str or None, optional): Filter description.
                The list of available filters can be found at
                https://ffmpeg.org/ffmpeg-filters.html
                Note that complex filters are not supported.

        """
        i = self.default_audio_stream if stream_index is None else stream_index
        if i is None:
            raise RuntimeError("There is no audio stream.")
        self._be.add_audio_stream(
            i,
            frames_per_chunk,
            buffer_chunk_size,
            filter_desc,
            decoder,
            decoder_option or {},
        )

    @_format_video_args
    def add_video_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        hw_accel: Optional[str] = None,
        filter_desc: Optional[str] = None,
    ):
        """Add output video stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            hw_accel (str or None, optional): {hw_accel}

            filter_desc (str or None, optional): Filter description.
                The list of available filters can be found at
                https://ffmpeg.org/ffmpeg-filters.html
                Note that complex filters are not supported.
        """
        i = self.default_video_stream if stream_index is None else stream_index
        if i is None:
            raise RuntimeError("There is no video stream.")
        self._be.add_video_stream(
            i,
            frames_per_chunk,
            buffer_chunk_size,
            filter_desc,
            decoder,
            decoder_option or {},
            hw_accel,
        )

    def remove_stream(self, i: int):
        """Remove an output stream.

        Args:
            i (int): Index of the output stream to be removed.
        """
        self._be.remove_stream(i)

    def process_packet(self, timeout: Optional[float] = None, backoff: float = 10.0) -> int:
        """Read the source media and process one packet.

        If a packet is read successfully, then the data in the packet will
        be decoded and passed to corresponding output stream processors.

        If the packet belongs to a source stream that is not connected to
        an output stream, then the data are discarded.

        When the source reaches EOF, then it triggers all the output stream
        processors to enter drain mode. All the output stream processors
        flush the pending frames.

        Args:
            timeout (float or None, optional): Timeout in milli seconds.

                This argument changes the retry behavior when it failed to
                process a packet due to the underlying media resource being
                temporarily unavailable.

                When using a media device such as a microphone, there are cases
                where the underlying buffer is not ready.
                Calling this function in such case would cause the system to report
                `EAGAIN (resource temporarily unavailable)`.

                * ``>=0``: Keep retrying until the given time passes.

                * ``0<``: Keep retrying forever.

                * ``None`` : No retrying and raise an exception immediately.

                Default: ``None``.

                Note:

                    The retry behavior is applicable only when the reason is the
                    unavailable resource. It is not invoked if the reason of failure is
                    other.

            backoff (float, optional): Time to wait before retrying in milli seconds.

                This option is effective only when `timeout` is effective. (not ``None``)

                When `timeout` is effective, this `backoff` controls how long the function
                should wait before retrying. Default: ``10.0``.

        Returns:
            int:
                ``0``
                A packet was processed properly. The caller can keep
                calling this function to buffer more frames.

                ``1``
                The streamer reached EOF. All the output stream processors
                flushed the pending frames. The caller should stop calling
                this method.
        """
        return self._be.process_packet(timeout, backoff)

    def process_all_packets(self):
        """Process packets until it reaches EOF."""
        self._be.process_all_packets()

    def is_buffer_ready(self) -> bool:
        """Returns true if all the output streams have at least one chunk filled."""
        return self._be.is_buffer_ready()

    def pop_chunks(self) -> Tuple[Optional[torch.Tensor]]:
        """Pop one chunk from all the output stream buffers.

        Returns:
            Tuple[Optional[Tensor]]:
                Buffer contents.
                If a buffer does not contain any frame, then `None` is returned instead.
        """
        return self._be.pop_chunks()

    def _fill_buffer(self, timeout: Optional[float], backoff: float) -> int:
        """Keep processing packets until all buffers have at least one chunk

        Returns:
            int:
                ``0``
                Packets are processed properly and buffers are
                ready to be popped once.

                ``1``
                The streamer reached EOF. All the output stream processors
                flushed the pending frames. The caller should stop calling
                this method.
        """
        while not self.is_buffer_ready():
            code = self.process_packet(timeout, backoff)
            if code != 0:
                return code
        return 0

    def stream(
        self, timeout: Optional[float] = None, backoff: float = 10.0
    ) -> Iterator[Tuple[Optional[torch.Tensor], ...]]:
        """Return an iterator that generates output tensors

        Arguments:
            timeout (float or None, optional): See
                :py:func:`~StreamReader.process_packet`. (Default: ``None``)

            backoff (float, optional): See
                :py:func:`~StreamReader.process_packet`. (Default: ``10.0``)

        Returns:
            Iterator[Tuple[Optional[torch.Tensor], ...]]:
                Iterator that yields a tuple of chunks that correspond to the output
                streams defined by client code.
                If an output stream is exhausted, then the chunk Tensor is substituted
                with ``None``.
                The iterator stops if all the output streams are exhausted.
        """
        if self.num_out_streams == 0:
            raise RuntimeError("No output stream is configured.")

        while True:
            if self._fill_buffer(timeout, backoff):
                break
            yield self.pop_chunks()

        while True:
            chunks = self.pop_chunks()
            if all(c is None for c in chunks):
                return
            yield chunks
