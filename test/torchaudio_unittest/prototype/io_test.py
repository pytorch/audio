import torch
from parameterized import parameterized
from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    TempDirMixin,
    get_asset_path,
    get_wav_data,
    save_wav,
    skipIfNoFFmpeg,
    nested_params,
    is_ffmpeg_available,
    get_image,
    save_image,
)

if is_ffmpeg_available():
    from torchaudio.prototype.io import (
        Streamer,
        SourceStream,
        SourceVideoStream,
        SourceAudioStream,
    )


def get_video_asset(file="nasa_13013.mp4"):
    return get_asset_path(file)


@skipIfNoFFmpeg
class StreamerInterfaceTest(TempDirMixin, TorchaudioTestCase):
    """Test suite for interface behaviors around Streamer"""

    def test_streamer_invalid_input(self):
        """Streamer constructor does not segfault but raise an exception when the input is invalid"""
        with self.assertRaises(RuntimeError):
            Streamer("foobar")

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
        """When invalid options are given, Streamer raises an exception with these keys"""
        options.update({k: k for k in invalid_keys})
        src = get_video_asset()
        with self.assertRaises(RuntimeError) as ctx:
            Streamer(src, option=options)
        assert all(f'"{k}"' in str(ctx.exception) for k in invalid_keys)

    def test_src_info(self):
        """`get_src_stream_info` properly fetches information"""
        s = Streamer(get_video_asset())
        assert s.num_src_streams == 6

        expected = [
            SourceVideoStream(
                media_type="video",
                codec="h264",
                codec_long_name="H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                format="yuv420p",
                bit_rate=71925,
                width=320,
                height=180,
                frame_rate=25.0,
            ),
            SourceAudioStream(
                media_type="audio",
                codec="aac",
                codec_long_name="AAC (Advanced Audio Coding)",
                format="fltp",
                bit_rate=72093,
                sample_rate=8000.0,
                num_channels=2,
            ),
            SourceStream(
                media_type="subtitle",
                codec="mov_text",
                codec_long_name="MOV text",
                format=None,
                bit_rate=None,
            ),
            SourceVideoStream(
                media_type="video",
                codec="h264",
                codec_long_name="H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                format="yuv420p",
                bit_rate=128783,
                width=480,
                height=270,
                frame_rate=29.97002997002997,
            ),
            SourceAudioStream(
                media_type="audio",
                codec="aac",
                codec_long_name="AAC (Advanced Audio Coding)",
                format="fltp",
                bit_rate=128837,
                sample_rate=16000.0,
                num_channels=2,
            ),
            SourceStream(
                media_type="subtitle",
                codec="mov_text",
                codec_long_name="MOV text",
                format=None,
                bit_rate=None,
            ),
        ]
        for i, exp in enumerate(expected):
            assert exp == s.get_src_stream_info(i)

    def test_src_info_invalid_index(self):
        """`get_src_stream_info` does not segfault but raise an exception when input is invalid"""
        s = Streamer(get_video_asset())
        for i in [-1, 6, 7, 8]:
            with self.assertRaises(IndexError):
                s.get_src_stream_info(i)

    def test_default_streams(self):
        """default stream is not None"""
        s = Streamer(get_video_asset())
        assert s.default_audio_stream is not None
        assert s.default_video_stream is not None

    def test_default_audio_stream_none(self):
        """default audio stream is None for video without audio"""
        s = Streamer(get_video_asset("nasa_13013_no_audio.mp4"))
        assert s.default_audio_stream is None

    def test_default_video_stream_none(self):
        """default video stream is None for video with only audio"""
        s = Streamer(get_video_asset("nasa_13013_no_video.mp4"))
        assert s.default_video_stream is None

    def test_num_out_stream(self):
        """num_out_streams gives the correct count of output streams"""
        s = Streamer(get_video_asset())
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
        s = Streamer(get_video_asset())
        s.add_basic_audio_stream(frames_per_chunk=-1, dtype=None)
        s.add_basic_audio_stream(frames_per_chunk=-1, sample_rate=8000)
        s.add_basic_audio_stream(frames_per_chunk=-1, dtype=torch.int16)

        sinfo = s.get_out_stream_info(0)
        assert sinfo.source_index == s.default_audio_stream
        assert sinfo.filter_description == ""

        sinfo = s.get_out_stream_info(1)
        assert sinfo.source_index == s.default_audio_stream
        assert "aresample=8000" in sinfo.filter_description

        sinfo = s.get_out_stream_info(2)
        assert sinfo.source_index == s.default_audio_stream
        assert "aformat=sample_fmts=s16" in sinfo.filter_description

    def test_basic_video_stream(self):
        """`add_basic_video_stream` constructs a correct filter."""
        s = Streamer(get_video_asset())
        s.add_basic_video_stream(frames_per_chunk=-1, format=None)
        s.add_basic_video_stream(frames_per_chunk=-1, width=3, height=5)
        s.add_basic_video_stream(frames_per_chunk=-1, frame_rate=7)
        s.add_basic_video_stream(frames_per_chunk=-1, format="BGR")

        sinfo = s.get_out_stream_info(0)
        assert sinfo.source_index == s.default_video_stream
        assert sinfo.filter_description == ""

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
        s = Streamer(get_video_asset())
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
        s = Streamer(get_video_asset())
        for i in range(-3, 3):
            with self.assertRaises(IndexError):
                s.remove_stream(i)

        s.add_audio_stream(frames_per_chunk=-1)
        for i in range(-3, 3):
            if i == 0:
                continue
            with self.assertRaises(IndexError):
                s.remove_stream(i)

    def test_process_packet(self):
        """`process_packet` method returns 0 while there is a packet in source stream"""
        s = Streamer(get_video_asset())
        # nasa_1013.mp3 contains 1023 packets.
        for _ in range(1023):
            code = s.process_packet()
            assert code == 0
        # now all the packets should be processed, so process_packet returns 1.
        code = s.process_packet()
        assert code == 1

    def test_pop_chunks_no_output_stream(self):
        """`pop_chunks` method returns empty list when there is no output stream"""
        s = Streamer(get_video_asset())
        assert s.pop_chunks() == []

    def test_pop_chunks_empty_buffer(self):
        """`pop_chunks` method returns None when a buffer is empty"""
        s = Streamer(get_video_asset())
        s.add_basic_audio_stream(frames_per_chunk=-1)
        s.add_basic_video_stream(frames_per_chunk=-1)
        assert s.pop_chunks() == [None, None]

    def test_pop_chunks_exhausted_stream(self):
        """`pop_chunks` method returns None when the source stream is exhausted"""
        s = Streamer(get_video_asset())
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
        s = Streamer(get_video_asset())
        with self.assertRaises(RuntimeError):
            next(s.stream())

    def test_stream_smoke_test(self):
        """`stream` streams chunks fine"""
        w, h = 256, 198
        s = Streamer(get_video_asset())
        s.add_basic_audio_stream(frames_per_chunk=2000, sample_rate=8000)
        s.add_basic_video_stream(frames_per_chunk=15, frame_rate=60, width=w, height=h)
        for i, (achunk, vchunk) in enumerate(s.stream()):
            assert achunk.shape == torch.Size([2000, 2])
            assert vchunk.shape == torch.Size([15, 3, h, w])
            if i >= 40:
                break


@skipIfNoFFmpeg
class StreamerAudioTest(TempDirMixin, TorchaudioTestCase):
    """Test suite for audio streaming"""

    def _get_reference_wav(self, sample_rate, channels_first=False, **kwargs):
        data = get_wav_data(**kwargs, normalize=False, channels_first=channels_first)
        path = self.get_temp_path("ref.wav")
        save_wav(path, data, sample_rate, channels_first=channels_first)
        return path, data

    def _test_wav(self, path, original, dtype):
        s = Streamer(path)
        s.add_basic_audio_stream(frames_per_chunk=-1, dtype=dtype)
        s.process_all_packets()
        (output,) = s.pop_chunks()
        self.assertEqual(original, output)

    @nested_params(
        ["int16", "uint8", "int32"],  # "float", "double", "int64"]
        [1, 2, 4, 8],
    )
    def test_basic_audio_stream(self, dtype, num_channels):
        """`basic_audio_stream` can load WAV file properly."""
        path, original = self._get_reference_wav(8000, dtype=dtype, num_channels=num_channels)

        # provide the matching dtype
        self._test_wav(path, original, getattr(torch, dtype))
        # use the internal dtype ffmpeg picks
        self._test_wav(path, original, None)

    @nested_params(
        ["int16", "uint8", "int32"],  # "float", "double", "int64"]
        [1, 2, 4, 8],
    )
    def test_audio_stream(self, dtype, num_channels):
        """`add_audio_stream` can apply filter"""
        path, original = self._get_reference_wav(8000, dtype=dtype, num_channels=num_channels)

        expected = torch.flip(original, dims=(0,))

        s = Streamer(path)
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
        path, original = self._get_reference_wav(1, dtype=dtype, num_channels=num_channels, num_frames=30)
        for t in range(10, 20):
            expected = original[t:, :]
            s = Streamer(path)
            s.add_audio_stream(frames_per_chunk=-1)
            s.seek(float(t))
            s.process_all_packets()
            (output,) = s.pop_chunks()
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
        path, original = self._get_reference_wav(
            8000, dtype="int16", num_channels=num_channels, num_frames=num_frames, channels_first=False
        )

        s = Streamer(path)
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
class StreamerImageTest(TempDirMixin, TorchaudioTestCase):
    def _get_reference_png(self, width: int, height: int, grayscale: bool):
        original = get_image(width, height, grayscale=grayscale)
        path = self.get_temp_path("ref.png")
        save_image(path, original, mode="L" if grayscale else "RGB")
        return path, original

    def _test_png(self, path, original, format=None):
        s = Streamer(path)
        s.add_basic_video_stream(frames_per_chunk=-1, format=format)
        s.process_all_packets()
        (output,) = s.pop_chunks()
        self.assertEqual(original, output)

    @nested_params([True, False])
    def test_png(self, grayscale):
        # TODO:
        # Add test with alpha channel (RGBA, ARGB, BGRA, ABGR)
        w, h = 32, 18
        path, original = self._get_reference_png(w, h, grayscale=grayscale)
        expected = original[None, ...]
        self._test_png(path, expected)

    @parameterized.expand(
        [
            ("hflip", 2),
            ("vflip", 1),
        ]
    )
    def test_png_effect(self, filter_desc, index):
        h, w = 111, 250
        path, original = self._get_reference_png(w, h, grayscale=False)
        expected = torch.flip(original, dims=(index,))[None, ...]

        s = Streamer(path)
        s.add_video_stream(frames_per_chunk=-1, filter_desc=filter_desc)
        s.process_all_packets()
        output = s.pop_chunks()[0]
        print("expected", expected)
        print("output", output)
        self.assertEqual(expected, output)
