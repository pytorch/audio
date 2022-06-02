import torch
from parameterized import parameterized, parameterized_class
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_image,
    get_wav_data,
    is_ffmpeg_available,
    nested_params,
    save_image,
    save_wav,
    skipIfNoFFmpeg,
    TempDirMixin,
    TorchaudioTestCase,
)

if is_ffmpeg_available():
    from torchaudio.io import (
        StreamReader,
        StreamReaderSourceAudioStream,
        StreamReaderSourceStream,
        StreamReaderSourceVideoStream,
    )


################################################################################
# Helper decorator and Mixin to duplicate the tests for fileobj
_media_source = parameterized_class(
    ("test_fileobj",),
    [(False,), (True,)],
    class_name_func=lambda cls, _, params: f'{cls.__name__}{"_fileobj" if params["test_fileobj"] else "_path"}',
)


class _MediaSourceMixin:
    def setUp(self):
        super().setUp()
        self.src = None

    def get_src(self, path):
        if not self.test_fileobj:
            return path
        if self.src is not None:
            raise ValueError("get_video_asset can be called only once.")

        self.src = open(path, "rb")
        return self.src

    def tearDown(self):
        if self.src is not None:
            self.src.close()
        super().tearDown()


################################################################################


@skipIfNoFFmpeg
@_media_source
class StreamReaderInterfaceTest(_MediaSourceMixin, TempDirMixin, TorchaudioTestCase):
    """Test suite for interface behaviors around StreamReader"""

    def get_src(self, file="nasa_13013.mp4"):
        return super().get_src(get_asset_path(file))

    def test_streamer_invalid_input(self):
        """StreamReader constructor does not segfault but raise an exception when the input is invalid"""
        with self.assertRaises(RuntimeError):
            StreamReader("foobar")

    @nested_params(
        [
            ("foo",),
            (
                "foo",
                "bar",
            ),
        ],
        [{}, {"sample_rate": "16000"}],
    )
    def test_streamer_invalide_option(self, invalid_keys, options):
        """When invalid options are given, StreamReader raises an exception with these keys"""
        options.update({k: k for k in invalid_keys})
        with self.assertRaises(RuntimeError) as ctx:
            StreamReader(self.get_src(), option=options)
        assert all(f'"{k}"' in str(ctx.exception) for k in invalid_keys)

    def test_src_info(self):
        """`get_src_stream_info` properly fetches information"""
        s = StreamReader(self.get_src())
        assert s.num_src_streams == 6

        expected = [
            StreamReaderSourceVideoStream(
                media_type="video",
                codec="h264",
                codec_long_name="H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                format="yuv420p",
                bit_rate=71925,
                num_frames=325,
                bits_per_sample=8,
                width=320,
                height=180,
                frame_rate=25.0,
            ),
            StreamReaderSourceAudioStream(
                media_type="audio",
                codec="aac",
                codec_long_name="AAC (Advanced Audio Coding)",
                format="fltp",
                bit_rate=72093,
                num_frames=103,
                bits_per_sample=0,
                sample_rate=8000.0,
                num_channels=2,
            ),
            StreamReaderSourceStream(
                media_type="subtitle",
                codec="mov_text",
                codec_long_name="MOV text",
                format=None,
                bit_rate=None,
                num_frames=None,
                bits_per_sample=None,
            ),
            StreamReaderSourceVideoStream(
                media_type="video",
                codec="h264",
                codec_long_name="H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                format="yuv420p",
                bit_rate=128783,
                num_frames=390,
                bits_per_sample=8,
                width=480,
                height=270,
                frame_rate=29.97002997002997,
            ),
            StreamReaderSourceAudioStream(
                media_type="audio",
                codec="aac",
                codec_long_name="AAC (Advanced Audio Coding)",
                format="fltp",
                bit_rate=128837,
                num_frames=205,
                bits_per_sample=0,
                sample_rate=16000.0,
                num_channels=2,
            ),
            StreamReaderSourceStream(
                media_type="subtitle",
                codec="mov_text",
                codec_long_name="MOV text",
                format=None,
                bit_rate=None,
                num_frames=None,
                bits_per_sample=None,
            ),
        ]
        output = [s.get_src_stream_info(i) for i in range(6)]
        assert expected == output

    def test_src_info_invalid_index(self):
        """`get_src_stream_info` does not segfault but raise an exception when input is invalid"""
        s = StreamReader(self.get_src())
        for i in [-1, 6, 7, 8]:
            with self.assertRaises(RuntimeError):
                s.get_src_stream_info(i)

    def test_default_streams(self):
        """default stream is not None"""
        s = StreamReader(self.get_src())
        assert s.default_audio_stream is not None
        assert s.default_video_stream is not None

    def test_default_audio_stream_none(self):
        """default audio stream is None for video without audio"""
        s = StreamReader(self.get_src("nasa_13013_no_audio.mp4"))
        assert s.default_audio_stream is None

    def test_default_video_stream_none(self):
        """default video stream is None for video with only audio"""
        s = StreamReader(self.get_src("nasa_13013_no_video.mp4"))
        assert s.default_video_stream is None

    def test_num_out_stream(self):
        """num_out_streams gives the correct count of output streams"""
        s = StreamReader(self.get_src())
        n, m = 6, 4
        for i in range(n):
            assert s.num_out_streams == i
            s.add_audio_stream(frames_per_chunk=-1)
        for i in range(m):
            assert s.num_out_streams == n - i
            s.remove_stream(0)
        for i in range(m):
            assert s.num_out_streams == n - m + i
            s.add_video_stream(frames_per_chunk=-1)
        for i in range(n):
            assert s.num_out_streams == n - i
            s.remove_stream(n - i - 1)
        assert s.num_out_streams == 0

    def test_basic_audio_stream(self):
        """`add_basic_audio_stream` constructs a correct filter."""
        s = StreamReader(self.get_src())
        s.add_basic_audio_stream(frames_per_chunk=-1, format=None)
        s.add_basic_audio_stream(frames_per_chunk=-1, sample_rate=8000)
        s.add_basic_audio_stream(frames_per_chunk=-1, format="s16p")

        sinfo = s.get_out_stream_info(0)
        assert sinfo.source_index == s.default_audio_stream
        assert sinfo.filter_description == "anull"

        sinfo = s.get_out_stream_info(1)
        assert sinfo.source_index == s.default_audio_stream
        assert "aresample=8000" in sinfo.filter_description

        sinfo = s.get_out_stream_info(2)
        assert sinfo.source_index == s.default_audio_stream
        assert "aformat=sample_fmts=s16" in sinfo.filter_description

    def test_basic_video_stream(self):
        """`add_basic_video_stream` constructs a correct filter."""
        s = StreamReader(self.get_src())
        s.add_basic_video_stream(frames_per_chunk=-1, format=None)
        s.add_basic_video_stream(frames_per_chunk=-1, width=3, height=5)
        s.add_basic_video_stream(frames_per_chunk=-1, frame_rate=7)
        s.add_basic_video_stream(frames_per_chunk=-1, format="bgr24")

        sinfo = s.get_out_stream_info(0)
        assert sinfo.source_index == s.default_video_stream
        assert sinfo.filter_description == "null"

        sinfo = s.get_out_stream_info(1)
        assert sinfo.source_index == s.default_video_stream
        assert "scale=width=3:height=5" in sinfo.filter_description

        sinfo = s.get_out_stream_info(2)
        assert sinfo.source_index == s.default_video_stream
        assert "fps=7" in sinfo.filter_description

        sinfo = s.get_out_stream_info(3)
        assert sinfo.source_index == s.default_video_stream
        assert "format=pix_fmts=bgr24" in sinfo.filter_description

    def test_remove_streams(self):
        """`remove_stream` removes the correct output stream"""
        s = StreamReader(self.get_src())
        s.add_basic_audio_stream(frames_per_chunk=-1, sample_rate=24000)
        s.add_basic_video_stream(frames_per_chunk=-1, width=16, height=16)
        s.add_basic_audio_stream(frames_per_chunk=-1, sample_rate=8000)

        sinfo = [s.get_out_stream_info(i) for i in range(3)]
        s.remove_stream(1)
        del sinfo[1]
        assert sinfo == [s.get_out_stream_info(i) for i in range(s.num_out_streams)]

        s.remove_stream(1)
        del sinfo[1]
        assert sinfo == [s.get_out_stream_info(i) for i in range(s.num_out_streams)]

        s.remove_stream(0)
        del sinfo[0]
        assert [] == [s.get_out_stream_info(i) for i in range(s.num_out_streams)]

    def test_remove_stream_invalid(self):
        """Attempt to remove invalid output streams raises IndexError"""
        s = StreamReader(self.get_src())
        for i in range(-3, 3):
            with self.assertRaises(RuntimeError):
                s.remove_stream(i)

        s.add_audio_stream(frames_per_chunk=-1)
        for i in range(-3, 3):
            if i == 0:
                continue
            with self.assertRaises(RuntimeError):
                s.remove_stream(i)

    def test_process_packet(self):
        """`process_packet` method returns 0 while there is a packet in source stream"""
        s = StreamReader(self.get_src())
        # nasa_1013.mp3 contains 1023 packets.
        for _ in range(1023):
            code = s.process_packet()
            assert code == 0
        # now all the packets should be processed, so process_packet returns 1.
        code = s.process_packet()
        assert code == 1

    def test_pop_chunks_no_output_stream(self):
        """`pop_chunks` method returns empty list when there is no output stream"""
        s = StreamReader(self.get_src())
        assert s.pop_chunks() == []

    def test_pop_chunks_empty_buffer(self):
        """`pop_chunks` method returns None when a buffer is empty"""
        s = StreamReader(self.get_src())
        s.add_basic_audio_stream(frames_per_chunk=-1)
        s.add_basic_video_stream(frames_per_chunk=-1)
        assert s.pop_chunks() == [None, None]

    def test_pop_chunks_exhausted_stream(self):
        """`pop_chunks` method returns None when the source stream is exhausted"""
        s = StreamReader(self.get_src())
        # video is 16.57 seconds.
        # audio streams per 10 second chunk
        # video streams per 20 second chunk
        # The first `pop_chunk` call should return 2 Tensors (10 second audio and 16.57 second video)
        # The second call should return 1 Tensor (6.57 second audio) and None.
        # After that, `pop_chunk` should keep returning None-s.
        s.add_basic_audio_stream(frames_per_chunk=100, sample_rate=10, buffer_chunk_size=3)
        s.add_basic_video_stream(frames_per_chunk=200, frame_rate=10, buffer_chunk_size=3)
        s.process_all_packets()
        chunks = s.pop_chunks()
        assert chunks[0] is not None
        assert chunks[1] is not None
        assert chunks[0].shape[0] == 100  # audio tensor contains 10 second chunk
        assert chunks[1].shape[0] < 200  # video tensor contains less than 20 second chunk
        chunks = s.pop_chunks()
        assert chunks[0] is not None
        assert chunks[1] is None
        assert chunks[0].shape[0] < 100  # audio tensor contains less than 10 second chunk
        for _ in range(10):
            chunks = s.pop_chunks()
            assert chunks[0] is None
            assert chunks[1] is None

    def test_stream_empty(self):
        """`stream` fails when no output stream is configured"""
        s = StreamReader(self.get_src())
        with self.assertRaises(RuntimeError):
            next(s.stream())

    def test_stream_smoke_test(self):
        """`stream` streams chunks fine"""
        w, h = 256, 198
        s = StreamReader(self.get_src())
        s.add_basic_audio_stream(frames_per_chunk=2000, sample_rate=8000)
        s.add_basic_video_stream(frames_per_chunk=15, frame_rate=60, width=w, height=h)
        for i, (achunk, vchunk) in enumerate(s.stream()):
            assert achunk.shape == torch.Size([2000, 2])
            assert vchunk.shape == torch.Size([15, 3, h, w])
            if i >= 40:
                break

    def test_seek(self):
        """Calling `seek` multiple times should not segfault"""
        s = StreamReader(self.get_src())
        for i in range(10):
            s.seek(i)
        for _ in range(0):
            s.seek(0)
        for i in range(10, 0, -1):
            s.seek(i)

    def test_seek_negative(self):
        """Calling `seek` with negative value should raise an exception"""
        s = StreamReader(self.get_src())
        with self.assertRaises(RuntimeError):
            s.seek(-1.0)


def _to_fltp(original):
    """Convert Tensor to float32 with value range [-1, 1]"""
    denom = {
        torch.uint8: 2**7,
        torch.int16: 2**15,
        torch.int32: 2**31,
    }[original.dtype]

    fltp = original.to(torch.float32)
    if original.dtype == torch.uint8:
        fltp -= 128
    fltp /= denom
    return fltp


@skipIfNoFFmpeg
@_media_source
class StreamReaderAudioTest(_MediaSourceMixin, TempDirMixin, TorchaudioTestCase):
    """Test suite for audio streaming"""

    def _get_reference_wav(self, sample_rate, channels_first=False, **kwargs):
        data = get_wav_data(**kwargs, normalize=False, channels_first=channels_first)
        path = self.get_temp_path("ref.wav")
        save_wav(path, data, sample_rate, channels_first=channels_first)
        return path, data

    def get_src(self, *args, **kwargs):
        path, data = self._get_reference_wav(*args, **kwargs)
        src = super().get_src(path)
        return src, data

    def _test_wav(self, src, original, fmt):
        s = StreamReader(src)
        s.add_basic_audio_stream(frames_per_chunk=-1, format=fmt)
        s.process_all_packets()
        (output,) = s.pop_chunks()
        self.assertEqual(original, output)

    @nested_params(
        ["int16", "uint8", "int32"],  # "float", "double", "int64"]
        [1, 2, 4, 8],
    )
    def test_basic_audio_stream(self, dtype, num_channels):
        """`basic_audio_stream` can load WAV file properly."""
        src, original = self.get_src(8000, dtype=dtype, num_channels=num_channels)

        fmt = {
            "uint8": "u8p",
            "int16": "s16p",
            "int32": "s32p",
        }[dtype]

        # provide the matching dtype
        self._test_wav(src, original, fmt=fmt)
        # use the internal dtype ffmpeg picks
        if self.test_fileobj:
            src.seek(0)
        self._test_wav(src, original, fmt=None)
        # convert to float32
        expected = _to_fltp(original)
        if self.test_fileobj:
            src.seek(0)
        self._test_wav(src, expected, fmt="fltp")

    @nested_params(
        ["int16", "uint8", "int32"],  # "float", "double", "int64"]
        [1, 2, 4, 8],
    )
    def test_audio_stream(self, dtype, num_channels):
        """`add_audio_stream` can apply filter"""
        src, original = self.get_src(8000, dtype=dtype, num_channels=num_channels)

        expected = torch.flip(original, dims=(0,))

        s = StreamReader(src)
        s.add_audio_stream(frames_per_chunk=-1, filter_desc="areverse")
        s.process_all_packets()
        (output,) = s.pop_chunks()
        self.assertEqual(expected, output)

    @nested_params(
        ["int16", "uint8", "int32"],  # "float", "double", "int64"]
        [1, 2, 4, 8],
    )
    def test_audio_seek(self, dtype, num_channels):
        """`seek` changes the position properly"""
        src, original = self.get_src(1, dtype=dtype, num_channels=num_channels, num_frames=30)

        for t in range(10, 20):
            expected = original[t:, :]
            if self.test_fileobj:
                src.seek(0)
            s = StreamReader(src)
            s.add_audio_stream(frames_per_chunk=-1)
            s.seek(float(t))
            s.process_all_packets()
            (output,) = s.pop_chunks()
            self.assertEqual(expected, output)

    def test_audio_seek_multiple(self):
        """Calling `seek` after streaming is started should change the position properly"""
        src, original = self.get_src(1, dtype="int16", num_channels=2, num_frames=30)

        s = StreamReader(src)
        s.add_audio_stream(frames_per_chunk=-1)

        ts = list(range(20)) + list(range(20, 0, -1)) + list(range(20))
        for t in ts:
            s.seek(float(t))
            s.process_all_packets()
            (output,) = s.pop_chunks()
            expected = original[t:, :]
            self.assertEqual(expected, output)

    @nested_params(
        [
            (18, 6, 3),  # num_frames is divisible by frames_per_chunk
            (18, 5, 4),  # num_frames is not divisible by frames_per_chunk
            (18, 32, 1),  # num_frames is shorter than frames_per_chunk
        ],
        [1, 2, 4, 8],
    )
    def test_audio_frames_per_chunk(self, frame_param, num_channels):
        """Different chunk parameter covers the source media properly"""
        num_frames, frames_per_chunk, buffer_chunk_size = frame_param
        src, original = self.get_src(
            8000, dtype="int16", num_channels=num_channels, num_frames=num_frames, channels_first=False
        )

        s = StreamReader(src)
        s.add_audio_stream(frames_per_chunk=frames_per_chunk, buffer_chunk_size=buffer_chunk_size)
        i, outputs = 0, []
        for (output,) in s.stream():
            expected = original[frames_per_chunk * i : frames_per_chunk * (i + 1), :]
            outputs.append(output)
            self.assertEqual(expected, output)
            i += 1
        assert i == num_frames // frames_per_chunk + (1 if num_frames % frames_per_chunk else 0)
        self.assertEqual(torch.cat(outputs, 0), original)


@skipIfNoFFmpeg
@_media_source
class StreamReaderImageTest(_MediaSourceMixin, TempDirMixin, TorchaudioTestCase):
    def _get_reference_png(self, width: int, height: int, grayscale: bool):
        original = get_image(width, height, grayscale=grayscale)
        path = self.get_temp_path("ref.png")
        save_image(path, original, mode="L" if grayscale else "RGB")
        return path, original

    def get_src(self, *args, **kwargs):
        path, data = self._get_reference_png(*args, **kwargs)
        src = super().get_src(path)
        return src, data

    def _test_png(self, path, original, format=None):
        s = StreamReader(path)
        s.add_basic_video_stream(frames_per_chunk=-1, format=format)
        s.process_all_packets()
        (output,) = s.pop_chunks()
        self.assertEqual(original, output)

    @nested_params([True, False])
    def test_png(self, grayscale):
        # TODO:
        # Add test with alpha channel (RGBA, ARGB, BGRA, ABGR)
        w, h = 32, 18
        src, original = self.get_src(w, h, grayscale=grayscale)
        expected = original[None, ...]
        self._test_png(src, expected)

    @parameterized.expand(
        [
            ("hflip", 2),
            ("vflip", 1),
        ]
    )
    def test_png_effect(self, filter_desc, index):
        h, w = 111, 250
        src, original = self.get_src(w, h, grayscale=False)
        expected = torch.flip(original, dims=(index,))[None, ...]

        s = StreamReader(src)
        s.add_video_stream(frames_per_chunk=-1, filter_desc=filter_desc)
        s.process_all_packets()
        output = s.pop_chunks()[0]
        print("expected", expected)
        print("output", output)
        self.assertEqual(expected, output)
