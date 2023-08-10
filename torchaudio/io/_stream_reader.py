from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterator, Optional, Tuple, TypeVar, Union

import torch
import torchaudio
from torch.utils._pytree import tree_map

if torchaudio._extension._FFMPEG_EXT is not None:
    _StreamReader = torchaudio._extension._FFMPEG_EXT.StreamReader
    _StreamReaderFileObj = torchaudio._extension._FFMPEG_EXT.StreamReaderFileObj


__all__ = [
    "StreamReader",
]


@dataclass
class SourceStream:
    """The metadata of a source stream, returned by :meth:`~torchaudio.io.StreamReader.get_src_stream_info`.

    This class is used when representing streams of media type other than `audio` or `video`.

    When source stream is `audio` or `video` type, :class:`SourceAudioStream` and
    :class:`SourceVideoStream`, which reports additional media-specific attributes,
    are used respectively.
    """

    media_type: str
    """The type of the stream.
    One of ``"audio"``, ``"video"``, ``"data"``, ``"subtitle"``, ``"attachment"`` and empty string.

    .. note::
       Only audio and video streams are supported for output.
    .. note::
       Still images, such as PNG and JPEG formats are reported as video.
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
    metadata: Dict[str, str]
    """Metadata attached to the source stream."""


@dataclass
class SourceAudioStream(SourceStream):
    """The metadata of an audio source stream, returned by :meth:`~torchaudio.io.StreamReader.get_src_stream_info`.

    This class is used when representing audio stream.

    In addition to the attributes reported by :class:`SourceStream`,
    the following attributes are reported.
    """

    sample_rate: float
    """Sample rate of the audio."""
    num_channels: int
    """Number of channels."""


@dataclass
class SourceVideoStream(SourceStream):
    """The metadata of a video source stream, returned by :meth:`~torchaudio.io.StreamReader.get_src_stream_info`.

    This class is used when representing video stream.

    In addition to the attributes reported by :class:`SourceStream`,
    the following attributes are reported.
    """

    width: int
    """Width of the video frame in pixel."""
    height: int
    """Height of the video frame in pixel."""
    frame_rate: float
    """Frame rate."""


def _parse_si(i):
    media_type = i.media_type
    if media_type == "audio":
        return SourceAudioStream(
            media_type=i.media_type,
            codec=i.codec_name,
            codec_long_name=i.codec_long_name,
            format=i.format,
            bit_rate=i.bit_rate,
            num_frames=i.num_frames,
            bits_per_sample=i.bits_per_sample,
            metadata=i.metadata,
            sample_rate=i.sample_rate,
            num_channels=i.num_channels,
        )
    if media_type == "video":
        return SourceVideoStream(
            media_type=i.media_type,
            codec=i.codec_name,
            codec_long_name=i.codec_long_name,
            format=i.format,
            bit_rate=i.bit_rate,
            num_frames=i.num_frames,
            bits_per_sample=i.bits_per_sample,
            metadata=i.metadata,
            width=i.width,
            height=i.height,
            frame_rate=i.frame_rate,
        )
    return SourceStream(
        media_type=i.media_type,
        codec=i.codec_name,
        codec_long_name=i.codec_long_name,
        format=None,
        bit_rate=None,
        num_frames=None,
        bits_per_sample=None,
        metadata=i.metadata,
    )


@dataclass
class OutputStream:
    """Output stream configured on :class:`StreamReader`,
    returned by :meth:`~torchaudio.io.StreamReader.get_out_stream_info`.
    """

    source_index: int
    """Index of the source stream that this output stream is connected."""
    filter_description: str
    """Description of filter graph applied to the source stream."""
    media_type: str
    """The type of the stream. ``"audio"`` or ``"video"``."""
    format: str
    """Media format. Such as ``"s16"`` and ``"yuv420p"``.

    Commonly found audio values are;

    - ``"u8"``, ``"u8p"``: Unsigned 8-bit unsigned interger.
    - ``"s16"``, ``"s16p"``: 16-bit signed integer.
    - ``"s32"``, ``"s32p"``: 32-bit signed integer.
    - ``"flt"``, ``"fltp"``: 32-bit floating-point.

    .. note::

       `p` at the end indicates the format is `planar`.
       Channels are grouped together instead of interspersed in memory."""


@dataclass
class OutputAudioStream(OutputStream):
    """Information about an audio output stream configured with
    :meth:`~torchaudio.io.StreamReader.add_audio_stream` or
    :meth:`~torchaudio.io.StreamReader.add_basic_audio_stream`.

    In addition to the attributes reported by :class:`OutputStream`,
    the following attributes are reported.
    """

    sample_rate: float
    """Sample rate of the audio."""
    num_channels: int
    """Number of channels."""


@dataclass
class OutputVideoStream(OutputStream):
    """Information about a video output stream configured with
    :meth:`~torchaudio.io.StreamReader.add_video_stream` or
    :meth:`~torchaudio.io.StreamReader.add_basic_video_stream`.

    In addition to the attributes reported by :class:`OutputStream`,
    the following attributes are reported.
    """

    width: int
    """Width of the video frame in pixel."""
    height: int
    """Height of the video frame in pixel."""
    frame_rate: float
    """Frame rate."""


def _parse_oi(i):
    media_type = i.media_type
    if media_type == "audio":
        return OutputAudioStream(
            source_index=i.source_index,
            filter_description=i.filter_description,
            media_type=i.media_type,
            format=i.format,
            sample_rate=i.sample_rate,
            num_channels=i.num_channels,
        )
    if media_type == "video":
        return OutputVideoStream(
            source_index=i.source_index,
            filter_description=i.filter_description,
            media_type=i.media_type,
            format=i.format,
            width=i.width,
            height=i.height,
            frame_rate=i.frame_rate,
        )
    raise ValueError(f"Unexpected media_type: {i.media_type}({i})")


def _get_afilter_desc(sample_rate: Optional[int], fmt: Optional[str], num_channels: Optional[int]):
    descs = []
    if sample_rate is not None:
        descs.append(f"aresample={sample_rate}")
    if fmt is not None or num_channels is not None:
        parts = []
        if fmt is not None:
            parts.append(f"sample_fmts={fmt}")
        if num_channels is not None:
            parts.append(f"channel_layouts={num_channels}c")
        descs.append(f"aformat={':'.join(parts)}")
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


# Base class for ChunkTensor
# Based off of TrivialTensorViaComposition
# https://github.com/albanD/subclass_zoo/blob/0eeb1d68fb59879029c610bc407f2997ae43ba0a/trivial_tensors.py#L83
class ChunkTensorBase(torch.Tensor):
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, _elem, *_):
        return super().__new__(cls, _elem)

    @classmethod
    def __torch_dispatch__(cls, func, _, args=(), kwargs=None):
        def unwrap(t):
            return t._elem if isinstance(t, cls) else t

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


@dataclass
class ChunkTensor(ChunkTensorBase):
    """Decoded media frames with metadata.

    The instance of this class represents the decoded video/audio frames with
    metadata, and the instance itself behave like :py:class:`~torch.Tensor`.

    Client codes can pass instance of this class as-if it's
    :py:class:`~torch.Tensor` class, or call the methods defined on
    :py:class:`~torch.Tensor` class.

    Example:
        >>> # Define input streams
        >>> reader = StreamReader(...)
        >>> reader.add_audio_stream(frames_per_chunk=4000, sample_rate=8000)
        >>> reader.add_video_stream(frames_per_chunk=7, frame_rate=28)
        >>> # Decode the streams and fetch frames
        >>> reader.fill_buffer()
        >>> audio_chunk, video_chunk = reader.pop_chunks()

        >>> # Access metadata
        >>> (audio_chunk.pts, video_chunks.pts)
        (0.0, 0.0)
        >>>
        >>> # The second time the PTS is different
        >>> reader.fill_buffer()
        >>> audio_chunk, video_chunk = reader.pop_chunks()
        >>> (audio_chunk.pts, video_chunks.pts)
        (0.5, 0.25)

        >>> # Call PyTorch ops on chunk
        >>> audio_chunk.shape
        torch.Size([4000, 2]
        >>> power = torch.pow(video_chunk, 2)
        >>>
        >>> # the result is a plain torch.Tensor class
        >>> type(power)
        <class 'torch.Tensor'>
        >>>
        >>> # Metadata is not available on the result
        >>> power.pts
        AttributeError: 'Tensor' object has no attribute 'pts'
    """

    # Keep it private for now
    _elem: torch.Tensor

    pts: float
    """Presentation time stamp of the first frame in the chunk.

    Unit: second.
    """


def _format_doc(**kwargs):
    def decorator(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj

    return decorator


_frames_per_chunk = """Number of frames returned as one chunk.
                If the source stream is exhausted before enough frames are buffered,
                then the chunk is returned as-is.

                Providing ``-1`` disables chunking and :py:func:`pop_chunks` method
                will concatenate all the buffered frames and return it."""

_buffer_chunk_size = """Internal buffer size.
                When the number of chunks buffered exceeds this number, old frames are
                dropped. For example, if ``frames_per_chunk`` is 5 and ``buffer_chunk_size`` is
                3, then frames older than ``15`` are dropped.
                Providing ``-1`` disables this behavior.

                Default: ``3``."""

_audio_stream_index = """The source audio stream index.
                If omitted, :py:attr:`default_audio_stream` is used."""


_video_stream_index = """The source video stream index.
                If omitted, :py:attr:`default_video_stream` is used."""

_decoder = """The name of the decoder to be used.
                When provided, use the specified decoder instead of the default one.

                To list the available decoders, please use
                :py:func:`~torchaudio.utils.ffmpeg_utils.get_audio_decoders` for audio, and
                :py:func:`~torchaudio.utils.ffmpeg_utils.get_video_decoders` for video.

                Default: ``None``."""

_decoder_option = """Options passed to decoder.
                Mapping from str to str. (Default: ``None``)

                To list decoder options for a decoder, you can use
                ``ffmpeg -h decoder=<DECODER>`` command.

                |

                In addition to decoder-specific options, you can also pass options related
                to multithreading. They are effective only if the decoder support them.
                If neither of them are provided, StreamReader defaults to single thread.

                ``"threads"``: The number of threads (in str).
                Providing the value ``"0"`` will let FFmpeg decides based on its heuristics.

                ``"thread_type"``: Which multithreading method to use.
                The valid values are ``"frame"`` or ``"slice"``.
                Note that each decoder supports different set of methods.
                If not provided, a default value is used.

                - ``"frame"``: Decode more than one frame at once.
                  Each thread handles one frame.
                  This will increase decoding delay by one frame per thread
                - ``"slice"``: Decode more than one part of a single frame at once.

                |
                """


_hw_accel = """Enable hardware acceleration.

                When video is decoded on CUDA hardware, for example
                `decoder="h264_cuvid"`, passing CUDA device indicator to `hw_accel`
                (i.e. `hw_accel="cuda:0"`) will make StreamReader place the resulting
                frames directly on the specified CUDA device as CUDA tensor.

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


InputStreamTypes = TypeVar("InputStream", bound=SourceStream)
OutputStreamTypes = TypeVar("OutputStream", bound=OutputStream)


@torchaudio._extension.fail_if_no_ffmpeg
class StreamReader:
    """Fetch and decode audio/video streams chunk by chunk.

    For the detailed usage of this class, please refer to the tutorial.

    Args:
        src (str, file-like object): The media source.
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

               https://ffmpeg.org/ffmpeg-formats.html#Demuxers

               Please use :py:func:`~torchaudio.utils.ffmpeg_utils.get_demuxers` to list the
               demultiplexers available in the current environment.

               For device access, the available values vary based on hardware (AV device) and
               software configuration (ffmpeg build).

               https://ffmpeg.org/ffmpeg-devices.html#Input-Devices

               Please use :py:func:`~torchaudio.utils.ffmpeg_utils.get_input_devices` to list
               the input devices available in the current environment.

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
        src: Union[str, BinaryIO],
        format: Optional[str] = None,
        option: Optional[Dict[str, str]] = None,
        buffer_size: int = 4096,
    ):
        if isinstance(src, str):
            self._be = _StreamReader(src, format, option)
        elif hasattr(src, "read"):
            self._be = _StreamReaderFileObj(src, format, option, buffer_size)
        else:
            raise ValueError("`src` must be either a string or file-like object.")

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

    def get_metadata(self) -> Dict[str, str]:
        """Get the metadata of the source media.

        Returns:
            dict
        """
        return self._be.get_metadata()

    def get_src_stream_info(self, i: int) -> InputStreamTypes:
        """Get the metadata of source stream

        Args:
            i (int): Stream index.
        Returns:
            InputStreamTypes:
                Information about the source stream.
                If the source stream is audio type, then
                :class:`~torchaudio.io._stream_reader.SourceAudioStream` is returned.
                If it is video type, then
                :class:`~torchaudio.io._stream_reader.SourceVideoStream` is returned.
                Otherwise :class:`~torchaudio.io._stream_reader.SourceStream` class is returned.
        """
        return _parse_si(self._be.get_src_stream_info(i))

    def get_out_stream_info(self, i: int) -> OutputStreamTypes:
        """Get the metadata of output stream

        Args:
            i (int): Stream index.
        Returns:
            OutputStreamTypes
                Information about the output stream.
                If the output stream is audio type, then
                :class:`~torchaudio.io._stream_reader.OutputAudioStream` is returned.
                If it is video type, then
                :class:`~torchaudio.io._stream_reader.OutputVideoStream` is returned.
        """
        info = self._be.get_out_stream_info(i)
        return _parse_oi(info)

    def seek(self, timestamp: float, mode: str = "precise"):
        """Seek the stream to the given timestamp [second]

        Args:
            timestamp (float): Target time in second.
            mode (str): Controls how seek is done.
                Valid choices are;

                * "key": Seek into the nearest key frame before the given timestamp.
                * "any": Seek into any frame (including non-key frames) before the given timestamp.
                * "precise": First seek into the nearest key frame before the given timestamp, then
                  decode frames until it reaches the closes frame to the given timestamp.

                Note:
                   All the modes invalidate and reset the internal state of decoder.
                   When using "any" mode and if it ends up seeking into non-key frame,
                   the image decoded may be invalid due to lack of key frame.
                   Using "precise" will workaround this issue by decoding frames from previous
                   key frame, but will be slower.
        """
        modes = {
            "key": 0,
            "any": 1,
            "precise": 2,
        }
        if mode not in modes:
            raise ValueError(f"The value of mode must be one of {list(modes.keys())}. Found: {mode}")
        self._be.seek(timestamp, modes[mode])

    @_format_audio_args
    def add_basic_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        *,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        format: Optional[str] = "fltp",
        sample_rate: Optional[int] = None,
        num_channels: Optional[int] = None,
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

            num_channels (int, or None, optional): If provided, change the number of channels.
        """
        self.add_audio_stream(
            frames_per_chunk,
            buffer_chunk_size,
            stream_index=stream_index,
            decoder=decoder,
            decoder_option=decoder_option,
            filter_desc=_get_afilter_desc(sample_rate, format, num_channels),
        )

    @_format_video_args
    def add_basic_video_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        *,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        format: Optional[str] = "rgb24",
        frame_rate: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        hw_accel: Optional[str] = None,
    ):
        """Add output video stream

        Args:
            frames_per_chunk (int): {frames_per_chunk}

            buffer_chunk_size (int, optional): {buffer_chunk_size}

            stream_index (int or None, optional): {stream_index}

            decoder (str or None, optional): {decoder}

            decoder_option (dict or None, optional): {decoder_option}

            format (str, optional): Change the format of image channels. Valid values are,

                - ``"rgb24"``: 8 bits * 3 channels (R, G, B)
                - ``"bgr24"``: 8 bits * 3 channels (B, G, R)
                - ``"yuv420p"``: 8 bits * 3 channels (Y, U, V)
                - ``"gray"``: 8 bits * 1 channels

                Default: ``"rgb24"``.

            frame_rate (int or None, optional): If provided, change the frame rate.

            width (int or None, optional): If provided, change the image width. Unit: Pixel.

            height (int or None, optional): If provided, change the image height. Unit: Pixel.

            hw_accel (str or None, optional): {hw_accel}
        """
        self.add_video_stream(
            frames_per_chunk,
            buffer_chunk_size,
            stream_index=stream_index,
            decoder=decoder,
            decoder_option=decoder_option,
            filter_desc=_get_vfilter_desc(frame_rate, width, height, format),
            hw_accel=hw_accel,
        )

    @_format_audio_args
    def add_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        *,
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
        *,
        stream_index: Optional[int] = None,
        decoder: Optional[str] = None,
        decoder_option: Optional[Dict[str, str]] = None,
        filter_desc: Optional[str] = None,
        hw_accel: Optional[str] = None,
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

    def pop_chunks(self) -> Tuple[Optional[ChunkTensor]]:
        """Pop one chunk from all the output stream buffers.

        Returns:
            Tuple[Optional[ChunkTensor]]:
                Buffer contents.
                If a buffer does not contain any frame, then `None` is returned instead.
        """
        ret = []
        for chunk in self._be.pop_chunks():
            if chunk is None:
                ret.append(None)
            else:
                ret.append(ChunkTensor(chunk.frames, chunk.pts))
        return ret

    def fill_buffer(self, timeout: Optional[float] = None, backoff: float = 10.0) -> int:
        """Keep processing packets until all buffers have at least one chunk

        Arguments:
            timeout (float or None, optional): See
                :py:func:`~StreamReader.process_packet`. (Default: ``None``)

            backoff (float, optional): See
                :py:func:`~StreamReader.process_packet`. (Default: ``10.0``)

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
        return self._be.fill_buffer(timeout, backoff)

    def stream(
        self, timeout: Optional[float] = None, backoff: float = 10.0
    ) -> Iterator[Tuple[Optional[ChunkTensor], ...]]:
        """Return an iterator that generates output tensors

        Arguments:
            timeout (float or None, optional): See
                :py:func:`~StreamReader.process_packet`. (Default: ``None``)

            backoff (float, optional): See
                :py:func:`~StreamReader.process_packet`. (Default: ``10.0``)

        Returns:
            Iterator[Tuple[Optional[ChunkTensor], ...]]:
                Iterator that yields a tuple of chunks that correspond to the output
                streams defined by client code.
                If an output stream is exhausted, then the chunk Tensor is substituted
                with ``None``.
                The iterator stops if all the output streams are exhausted.
        """
        if self.num_out_streams == 0:
            raise RuntimeError("No output stream is configured.")

        while True:
            if self.fill_buffer(timeout, backoff):
                break
            yield self.pop_chunks()

        while True:
            chunks = self.pop_chunks()
            if all(c is None for c in chunks):
                return
            yield chunks
