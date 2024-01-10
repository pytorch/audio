import io
from typing import Iterator, List, Optional

import torch
from torch import Tensor

from torio.io._streaming_media_decoder import _get_afilter_desc, StreamingMediaDecoder as StreamReader
from torio.io._streaming_media_encoder import CodecConfig, StreamingMediaEncoder as StreamWriter


class _StreamingIOBuffer:
    """Streaming Bytes IO buffer. Data are dropped when read."""

    def __init__(self):
        self._buffer: List(bytes) = []

    def write(self, b: bytes):
        if b:
            self._buffer.append(b)
        return len(b)

    def pop(self, n):
        """Pop the oldest byte string. It does not necessary return the requested amount"""
        if not self._buffer:
            return b""
        if len(self._buffer[0]) <= n:
            return self._buffer.pop(0)
        ret = self._buffer[0][:n]
        self._buffer[0] = self._buffer[0][n:]
        return ret


def _get_sample_fmt(dtype: torch.dtype):
    types = {
        torch.uint8: "u8",
        torch.int16: "s16",
        torch.int32: "s32",
        torch.float32: "flt",
        torch.float64: "dbl",
    }
    if dtype not in types:
        raise ValueError(f"Unsupported dtype is provided {dtype}. Supported dtypes are: {types.keys()}")
    return types[dtype]


class _AudioStreamingEncoder:
    """Given a waveform, encode on-demand and return bytes"""

    def __init__(
        self,
        src: Tensor,
        sample_rate: int,
        effect: str,
        muxer: str,
        encoder: Optional[str],
        codec_config: Optional[CodecConfig],
        frames_per_chunk: int,
    ):
        self.src = src
        self.buffer = _StreamingIOBuffer()
        self.writer = StreamWriter(self.buffer, format=muxer)
        self.writer.add_audio_stream(
            num_channels=src.size(1),
            sample_rate=sample_rate,
            format=_get_sample_fmt(src.dtype),
            encoder=encoder,
            filter_desc=effect,
            codec_config=codec_config,
        )
        self.writer.open()
        self.fpc = frames_per_chunk

        # index on the input tensor (along time-axis)
        # we use -1 to indicate that we finished iterating the tensor and
        # the writer is closed.
        self.i_iter = 0

    def read(self, n):
        while not self.buffer._buffer and self.i_iter >= 0:
            self.writer.write_audio_chunk(0, self.src[self.i_iter : self.i_iter + self.fpc])
            self.i_iter += self.fpc
            if self.i_iter >= self.src.size(0):
                self.writer.flush()
                self.writer.close()
                self.i_iter = -1
        return self.buffer.pop(n)


def _encode(
    src: Tensor,
    sample_rate: int,
    effect: str,
    muxer: str,
    encoder: Optional[str],
    codec_config: Optional[CodecConfig],
):
    buffer = io.BytesIO()
    writer = StreamWriter(buffer, format=muxer)
    writer.add_audio_stream(
        num_channels=src.size(1),
        sample_rate=sample_rate,
        format=_get_sample_fmt(src.dtype),
        encoder=encoder,
        filter_desc=effect,
        codec_config=codec_config,
    )
    with writer.open():
        writer.write_audio_chunk(0, src)
    buffer.seek(0)
    return buffer


def _get_muxer(dtype: torch.dtype):
    # TODO: check if this works in Windows.
    types = {
        torch.uint8: "u8",
        torch.int16: "s16le",
        torch.int32: "s32le",
        torch.float32: "f32le",
        torch.float64: "f64le",
    }
    if dtype not in types:
        raise ValueError(f"Unsupported dtype is provided {dtype}. Supported dtypes are: {types.keys()}")
    return types[dtype]


class AudioEffector:
    """Apply various filters and/or codecs to waveforms.

    .. versionadded:: 2.1

    Args:
        effect (str or None, optional): Filter expressions or ``None`` to apply no filter.
            See https://ffmpeg.org/ffmpeg-filters.html#Audio-Filters for the
            details of filter syntax.

        format (str or None, optional): When provided, encode the audio into the
            corresponding format. Default: ``None``.

        encoder (str or None, optional): When provided, override the encoder used
            by the ``format``. Default: ``None``.

        codec_config (CodecConfig or None, optional): When provided, configure the encoding codec.
            Should be provided in conjunction with ``format`` option.

        pad_end (bool, optional): When enabled, and if the waveform becomes shorter after applying
            effects/codec, then pad the end with silence.

    Example - Basic usage
        To use ``AudioEffector``, first instantiate it with a set of
        ``effect`` and ``format``.

        >>> # instantiate the effector
        >>> effector = AudioEffector(effect=..., format=...)

        Then, use :py:meth:`~AudioEffector.apply` or :py:meth:`~AudioEffector.stream`
        method to apply them.

        >>> # Apply the effect to the whole waveform
        >>> applied = effector.apply(waveform, sample_rate)

        >>> # Apply the effect chunk-by-chunk
        >>> for chunk in effector.stream(waveform, sample_rate):
        >>>    ...

    Example - Applying effects
        Please refer to
        https://ffmpeg.org/ffmpeg-filters.html#Filtergraph-description
        for the overview of filter description, and
        https://ffmpeg.org/ffmpeg-filters.html#toc-Audio-Filters
        for the list of available filters.

        Tempo - https://ffmpeg.org/ffmpeg-filters.html#atempo

        >>> AudioEffector(effect="atempo=1.5")

        Echo - https://ffmpeg.org/ffmpeg-filters.html#aecho

        >>> AudioEffector(effect="aecho=0.8:0.88:60:0.4")

        Flanger - https://ffmpeg.org/ffmpeg-filters.html#flanger

        >>> AudioEffector(effect="aflanger")

        Vibrato - https://ffmpeg.org/ffmpeg-filters.html#vibrato

        >>> AudioEffector(effect="vibrato")

        Tremolo - https://ffmpeg.org/ffmpeg-filters.html#tremolo

        >>> AudioEffector(effect="vibrato")

        You can also apply multiple effects at once.

        >>> AudioEffector(effect="")

    Example - Applying codec
        One can apply codec using ``format`` argument. ``format`` can be
        audio format or container format. If the container format supports
        multiple encoders, you can specify it with ``encoder`` argument.

        Wav format
        (no compression is applied but samples are converted to
        16-bit signed integer)

        >>> AudioEffector(format="wav")

        Ogg format with default encoder

        >>> AudioEffector(format="ogg")

        Ogg format with vorbis

        >>> AudioEffector(format="ogg", encoder="vorbis")

        Ogg format with opus

        >>> AudioEffector(format="ogg", encoder="opus")

        Webm format with opus

        >>> AudioEffector(format="webm", encoder="opus")

    Example - Applying codec with configuration
        Reference: https://trac.ffmpeg.org/wiki/Encode/MP3

        MP3 with default config

        >>> AudioEffector(format="mp3")

        MP3 with variable bitrate

        >>> AudioEffector(format="mp3", codec_config=CodecConfig(qscale=5))

        MP3 with constant bitrate

        >>> AudioEffector(format="mp3", codec_config=CodecConfig(bit_rate=32_000))
    """

    def __init__(
        self,
        effect: Optional[str] = None,
        format: Optional[str] = None,
        *,
        encoder: Optional[str] = None,
        codec_config: Optional[CodecConfig] = None,
        pad_end: bool = True,
    ):
        if format is None:
            if encoder is not None or codec_config is not None:
                raise ValueError("`encoder` and/or `condec_config` opions are provided without `format` option.")
        self.effect = effect
        self.format = format
        self.encoder = encoder
        self.codec_config = codec_config
        self.pad_end = pad_end

    def _get_reader(self, waveform, sample_rate, output_sample_rate, frames_per_chunk=None):
        num_frames, num_channels = waveform.shape

        if self.format is not None:
            muxer = self.format
            encoder = self.encoder
            option = {}
            # Some formats are headerless, so need to provide these infomation.
            if self.format == "mulaw":
                option = {"sample_rate": f"{sample_rate}", "channels": f"{num_channels}"}

        else:  # PCM
            muxer = _get_muxer(waveform.dtype)
            encoder = None
            option = {"sample_rate": f"{sample_rate}", "channels": f"{num_channels}"}

        if frames_per_chunk is None:
            src = _encode(waveform, sample_rate, self.effect, muxer, encoder, self.codec_config)
        else:
            src = _AudioStreamingEncoder(
                waveform, sample_rate, self.effect, muxer, encoder, self.codec_config, frames_per_chunk
            )

        output_sr = sample_rate if output_sample_rate is None else output_sample_rate
        filter_desc = _get_afilter_desc(output_sr, _get_sample_fmt(waveform.dtype), num_channels)
        if self.pad_end:
            filter_desc = f"{filter_desc},apad=whole_len={num_frames}"

        reader = StreamReader(src, format=muxer, option=option)
        reader.add_audio_stream(frames_per_chunk or -1, -1, filter_desc=filter_desc)
        return reader

    def apply(self, waveform: Tensor, sample_rate: int, output_sample_rate: Optional[int] = None) -> Tensor:
        """Apply the effect and/or codecs to the whole tensor.

        Args:
            waveform (Tensor): The input waveform. Shape: ``(time, channel)``
            sample_rate (int): Sample rate of the input waveform.
            output_sample_rate (int or None, optional): Output sample rate.
                If provided, override the output sample rate.
                Otherwise, the resulting tensor is resampled to have
                the same sample rate as the input.
                Default: ``None``.

        Returns:
            Tensor:
                Resulting Tensor. Shape: ``(time, channel)``. The number of frames
                could be different from that of the input.
        """
        if waveform.ndim != 2:
            raise ValueError(f"Expected the input waveform to be 2D. Found: {waveform.ndim}")

        if waveform.numel() == 0:
            return waveform

        reader = self._get_reader(waveform, sample_rate, output_sample_rate)
        reader.process_all_packets()
        (applied,) = reader.pop_chunks()
        return Tensor(applied)

    def stream(
        self, waveform: Tensor, sample_rate: int, frames_per_chunk: int, output_sample_rate: Optional[int] = None
    ) -> Iterator[Tensor]:
        """Apply the effect and/or codecs to the given tensor chunk by chunk.

        Args:
            waveform (Tensor): The input waveform. Shape: ``(time, channel)``
            sample_rate (int): Sample rate of the waveform.
            frames_per_chunk (int): The number of frames to return at a time.
            output_sample_rate (int or None, optional): Output sample rate.
                If provided, override the output sample rate.
                Otherwise, the resulting tensor is resampled to have
                the same sample rate as the input.
                Default: ``None``.

        Returns:
            Iterator[Tensor]:
                Series of processed chunks. Shape: ``(time, channel)``, where the
                the number of frames matches ``frames_per_chunk`` except the
                last chunk, which could be shorter.
        """
        if waveform.ndim != 2:
            raise ValueError(f"Expected the input waveform to be 2D. Found: {waveform.ndim}")

        if waveform.numel() == 0:
            return waveform

        reader = self._get_reader(waveform, sample_rate, output_sample_rate, frames_per_chunk)
        for (applied,) in reader.stream():
            yield Tensor(applied)
