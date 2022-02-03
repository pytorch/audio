from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterator

import torch
import torchaudio


@dataclass
class SourceStream:
    """SourceStream()

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
    """Short name of the codec. Such as `pcm_s16le` and `h264`."""
    codec_long_name: str
    """Detailed name of the codec.

    Such as `"PCM signed 16-bit little-endian"` and `"H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10"`.
    """
    format: Optional[str]
    """Media format. Such as `s16` and `yuv420p`.

    Commonly found audio values are;

    - `u8`, `u8p`: Unsigned 8-bit unsigned interger.
    - `s16`, `s16p`: 16-bit signed integer.
    - `s32`, `s32p`: 32-bit signed integer.
    - `flt`, `fltp`: 32-bit floating-point.

    .. note::

       `p` at the end indicates the format is `planar`.
       Channels are grouped together instead of interspersed in memory.
    """
    bit_rate: Optional[int]
    """Bit rate of the stream in bits-per-second.
    This is an estimated values based on the initial few frames of the stream.
    For container formats and variable bit rate, it can be 0.
    """


@dataclass
class SourceAudioStream(SourceStream):
    """SourceAudioStream()

    The metadata of an audio source stream.

    In addition to the attributes reported by :py:func:`SourceStream`,
    when the source stream is audio type, then the following additional attributes
    are reported.
    """

    sample_rate: float
    """Sample rate of the audio."""
    num_channels: int
    """Number of channels."""


@dataclass
class SourceVideoStream(SourceStream):
    """SourceVideoStream()

    The metadata of a video source stream.

    In addition to the attributes reported by :py:func:`SourceStream`,
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
# - AUDIO
_SAMPLE_RATE = 5
_NUM_CHANNELS = 6
# - VIDEO
_WIDTH = 7
_HEIGHT = 8
_FRAME_RATE = 9


def _parse_si(i):
    media_type = i[_MEDIA_TYPE]
    codec_name = i[_CODEC]
    codec_long_name = i[_CODEC_LONG]
    if media_type == "audio":
        return SourceAudioStream(
            media_type,
            codec_name,
            codec_long_name,
            i[_FORMAT],
            i[_BIT_RATE],
            i[_SAMPLE_RATE],
            i[_NUM_CHANNELS],
        )
    if media_type == "video":
        return SourceVideoStream(
            media_type,
            codec_name,
            codec_long_name,
            i[_FORMAT],
            i[_BIT_RATE],
            i[_WIDTH],
            i[_HEIGHT],
            i[_FRAME_RATE],
        )
    return SourceStream(media_type, codec_name, codec_long_name, None, None)


@dataclass
class OutputStream:
    """OutputStream()

    Output stream configured on :py:class:`Streamer`.
    """

    source_index: int
    """Index of the source stream that this output stream is connected."""
    filter_description: str
    """Description of filter graph applied to the source stream."""


def _parse_oi(i):
    return OutputStream(i[0], i[1])


class Streamer:
    """Fetch and decode audio/video streams chunk by chunk.

    For the detailed usage of this class, please refer to the tutorial.

    Args:
        src (str): Source. Can be a file path, URL, device identifier or filter expression.
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
    """

    def __init__(
        self,
        src: str,
        format: Optional[str] = None,
        option: Optional[Dict[str, str]] = None,
    ):
        self._s = torch.ops.torchaudio.ffmpeg_streamer_init(src, format, option)
        i = torch.ops.torchaudio.ffmpeg_streamer_find_best_audio_stream(self._s)
        self._i_audio = None if i < 0 else i
        i = torch.ops.torchaudio.ffmpeg_streamer_find_best_video_stream(self._s)
        self._i_video = None if i < 0 else i

    @property
    def num_src_streams(self):
        """Number of streams found in the provided media source.

        :type: int
        """
        return torch.ops.torchaudio.ffmpeg_streamer_num_src_streams(self._s)

    @property
    def num_out_streams(self):
        """Number of output streams configured by client code.

        :type: int
        """
        return torch.ops.torchaudio.ffmpeg_streamer_num_out_streams(self._s)

    @property
    def default_audio_stream(self):
        """The index of default audio stream. ``None`` if there is no audio stream

        :type: Optional[int]
        """
        return self._i_audio

    @property
    def default_video_stream(self):
        """The index of default video stream. ``None`` if there is no video stream

        :type: Optional[int]
        """
        return self._i_video

    def get_src_stream_info(self, i: int) -> torchaudio.prototype.io.SourceStream:
        """Get the metadata of source stream

        Args:
            i (int): Stream index.
        Returns:
            SourceStream
        """
        return _parse_si(torch.ops.torchaudio.ffmpeg_streamer_get_src_stream_info(self._s, i))

    def get_out_stream_info(self, i: int) -> torchaudio.prototype.io.OutputStream:
        """Get the metadata of output stream

        Args:
            i (int): Stream index.
        Returns:
            OutputStream
        """
        return _parse_oi(torch.ops.torchaudio.ffmpeg_streamer_get_out_stream_info(self._s, i))

    def seek(self, timestamp: float):
        """Seek the stream to the given timestamp [second]

        Args:
            timestamp (float): Target time in second.
        """
        torch.ops.torchaudio.ffmpeg_streamer_seek(self._s, timestamp)

    def add_basic_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        sample_rate: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Add output audio stream

        Args:
            frames_per_chunk (int): Number of frames returned by Streamer as a chunk.
                If the source stream is exhausted before enough frames are buffered,
                then the chunk is returned as-is.

            buffer_chunk_size (int, optional): Internal buffer size.
                When this many chunks are created, but
                client code does not pop the chunk, if a new frame comes in, the old
                chunk is dropped.

            stream_index (int or None, optional): The source audio stream index.
                If omitted, :py:attr:`default_audio_stream` is used.

            sample_rate (int or None, optional): If provided, resample the audio.

            dtype (torch.dtype, optional): If not ``None``, change the output sample precision.
                If floating point, then the sample value range is
                `[-1, 1]`.
        """
        i = self.default_audio_stream if stream_index is None else stream_index
        torch.ops.torchaudio.ffmpeg_streamer_add_basic_audio_stream(
            self._s, i, frames_per_chunk, buffer_chunk_size, sample_rate, dtype
        )

    def add_basic_video_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        frame_rate: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: str = "RGB",
    ):
        """Add output video stream

        Args:
            frames_per_chunk (int): Number of frames returned by Streamer as a chunk.
                If the source stream is exhausted before enough frames are buffered,
                then the chunk is returned as-is.

            buffer_chunk_size (int, optional): Internal buffer size.
                When this many chunks are created, but
                client code does not pop the chunk, if a new frame comes in, the old
                chunk is dropped.

            stream_index (int or None, optional): The source video stream index.
                If omitted, :py:attr:`default_video_stream` is used.

            frame_rate (int or None, optional): If provided, change the frame rate.

            width (int or None, optional): If provided, change the image width. Unit: Pixel.
            height (int or None, optional): If provided, change the image height. Unit: Pixel.
            format (str, optional): Change the format of image channels. Valid values are,

                - `RGB`: 8 bits * 3 channels
                - `BGR`: 8 bits * 3 channels
                - `GRAY`: 8 bits * 1 channels
        """
        i = self.default_video_stream if stream_index is None else stream_index
        torch.ops.torchaudio.ffmpeg_streamer_add_basic_video_stream(
            self._s,
            i,
            frames_per_chunk,
            buffer_chunk_size,
            frame_rate,
            width,
            height,
            format,
        )

    def add_audio_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        filter_desc: Optional[str] = None,
    ):
        """Add output audio stream

        Args:
            frames_per_chunk (int): Number of frames returned by Streamer as a chunk.
                If the source stream is exhausted before enough frames are buffered,
                then the chunk is returned as-is.

            buffer_chunk_size (int, optional): Internal buffer size.
                When this many chunks are created, but
                client code does not pop the chunk, if a new frame comes in, the old
                chunk is dropped.

            stream_index (int or None, optional): The source audio stream index.
                If omitted, :py:attr:`default_audio_stream` is used.

            filter_desc (str or None, optional): Filter description.
                The list of available filters can be found at
                https://ffmpeg.org/ffmpeg-filters.html
                Note that complex filters are not supported.
        """
        i = self.default_audio_stream if stream_index is None else stream_index
        torch.ops.torchaudio.ffmpeg_streamer_add_audio_stream(
            self._s, i, frames_per_chunk, buffer_chunk_size, filter_desc
        )

    def add_video_stream(
        self,
        frames_per_chunk: int,
        buffer_chunk_size: int = 3,
        stream_index: Optional[int] = None,
        filter_desc: Optional[str] = None,
    ):
        """Add output video stream

        Args:
            frames_per_chunk (int): Number of frames returned by Streamer as a chunk.
                If the source stream is exhausted before enough frames are buffered,
                then the chunk is returned as-is.

            buffer_chunk_size (int): Internal buffer size.
                When this many chunks are created, but
                client code does not pop the chunk, if a new frame comes in, the old
                chunk is dropped.

            stream_index (int or None, optional): The source video stream index.
                If omitted, :py:attr:`default_video_stream` is used.

            filter_desc (str or None, optional): Filter description.
                The list of available filters can be found at
                https://ffmpeg.org/ffmpeg-filters.html
                Note that complex filters are not supported.
        """
        i = self.default_video_stream if stream_index is None else stream_index
        torch.ops.torchaudio.ffmpeg_streamer_add_video_stream(
            self._s, i, frames_per_chunk, buffer_chunk_size, filter_desc
        )

    def remove_stream(self, i: int):
        """Remove an output stream.

        Args:
            i (int): Index of the output stream to be removed.
        """
        torch.ops.torchaudio.ffmpeg_streamer_remove_stream(self._s, i)

    def process_packet(self) -> int:
        """Read the source media and process one packet.

        The data in the packet will be decoded and passed to corresponding
        output stream processors.

        If the packet belongs to a source stream that is not connected to
        an output stream, then the data are discarded.

        When the source reaches EOF, then it triggers all the output stream
        processors to enter drain mode. All the output stream processors
        flush the pending frames.

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
        return torch.ops.torchaudio.ffmpeg_streamer_process_packet(self._s)

    def process_all_packets(self):
        """Process packets until it reaches EOF."""
        torch.ops.torchaudio.ffmpeg_streamer_process_all_packets(self._s)

    def is_buffer_ready(self) -> bool:
        """Returns true if all the output streams have at least one chunk filled."""
        return torch.ops.torchaudio.ffmpeg_streamer_is_buffer_ready(self._s)

    def pop_chunks(self) -> Tuple[Optional[torch.Tensor]]:
        """Pop one chunk from all the output stream buffers.

        Returns
            Tuple[Optional[Tensor]]:
                Buffer contents.
                If a buffer does not contain any frame, then `None` is returned instead.
        """
        return torch.ops.torchaudio.ffmpeg_streamer_pop_chunks(self._s)

    def fill_buffer(self) -> int:
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
            for _ in range(3):
                code = self.process_packet()
                if code != 0:
                    return code
        return 0

    def stream(self) -> Iterator[Tuple[Optional[torch.Tensor]]]:
        """Return an iterator that generates output tensors"""
        if self.num_out_streams == 0:
            raise RuntimeError("No output stream is configured.")

        while True:
            if self.fill_buffer():
                break
            yield self.pop_chunks()

        while True:
            chunks = self.pop_chunks()
            if all(c is None for c in chunks):
                return
            yield chunks
