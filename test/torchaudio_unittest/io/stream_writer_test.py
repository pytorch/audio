import io
import math

import torch
import torchaudio

from parameterized import parameterized, parameterized_class
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_sinusoid,
    is_ffmpeg_available,
    nested_params,
    rgb_to_yuv_ccir,
    skipIfNoFFmpeg,
    skipIfNoModule,
    TempDirMixin,
    TorchaudioTestCase,
)

if is_ffmpeg_available():
    from torchaudio.io import StreamReader, StreamWriter


def get_audio_chunk(fmt, sample_rate, num_channels):
    path = get_asset_path("nasa_13013.mp4")
    s = StreamReader(path)
    for _ in range(num_channels):
        s.add_basic_audio_stream(-1, -1, format=fmt, sample_rate=sample_rate)
    s.stream()
    s.process_all_packets()
    chunks = [chunk[:, :1] for chunk in s.pop_chunks()]
    return torch.cat(chunks, 1)


def get_video_chunk(fmt, frame_rate, *, width, height):
    path = get_asset_path("nasa_13013_no_audio.mp4")
    s = StreamReader(path)
    s.add_basic_video_stream(-1, -1, format=fmt, frame_rate=frame_rate, width=width, height=height)
    s.stream()
    s.process_all_packets()
    (chunk,) = s.pop_chunks()
    return chunk


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

    def get_dst(self, path):
        if not self.test_fileobj:
            return path
        if self.src is not None:
            raise ValueError("get_dst can be called only once.")

        self.src = open(path, "wb")
        return self.src

    def tearDown(self):
        if self.src is not None:
            self.src.flush()
            self.src.close()
        super().tearDown()


################################################################################


@skipIfNoFFmpeg
@_media_source
class StreamWriterInterfaceTest(_MediaSourceMixin, TempDirMixin, TorchaudioTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torchaudio.utils.ffmpeg_utils.set_log_level(32)

    @classmethod
    def tearDownClass(cls):
        torchaudio.utils.ffmpeg_utils.set_log_level(8)
        super().tearDownClass()

    def get_dst(self, path):
        return super().get_dst(self.get_temp_path(path))

    def get_buf(self, path):
        with open(self.get_temp_path(path), "rb") as fileobj:
            return fileobj.read()

    def test_unopened_error(self):
        """If dst is not opened when attempting to write data, runtime error should be raised"""
        path = self.get_dst("test.mp4")
        s = StreamWriter(path, format="mp4")
        s.set_metadata(metadata={"artist": "torchaudio", "title": self.id()})
        s.add_audio_stream(sample_rate=16000, num_channels=2)
        s.add_video_stream(frame_rate=30, width=16, height=16)

        dummy = torch.zeros((3, 2))
        with self.assertRaises(RuntimeError):
            s.write_audio_chunk(0, dummy)

        dummy = torch.zeros((3, 3, 16, 16))
        with self.assertRaises(RuntimeError):
            s.write_video_chunk(1, dummy)

    @skipIfNoModule("tinytag")
    def test_metadata_overwrite(self):
        """When set_metadata is called multiple times, only entries from the last call are saved"""
        from tinytag import TinyTag

        src_fmt = "s16"
        sample_rate = 8000
        num_channels = 1

        dst = self.get_dst("test.mp3")
        s = StreamWriter(dst, format="mp3")
        s.set_metadata(metadata={"artist": "torchaudio", "title": "foo"})
        s.set_metadata(metadata={"title": self.id()})
        s.add_audio_stream(sample_rate, num_channels, format=src_fmt)

        chunk = get_audio_chunk(src_fmt, sample_rate, num_channels)
        with s.open():
            s.write_audio_chunk(0, chunk)

        path = self.get_temp_path("test.mp3")
        tag = TinyTag.get(path)
        assert tag.artist is None
        assert tag.title == self.id()

    @nested_params(
        # Note: "s64" causes UB (left shift of 1 by 63 places cannot be represented in type 'long')
        # thus it's omitted.
        ["u8", "s16", "s32", "flt", "dbl"],
        [8000, 16000, 44100],
        [1, 2, 4],
    )
    def test_valid_audio_muxer_and_codecs_wav(self, src_fmt, sample_rate, num_channels):
        """Tensor of various dtypes can be saved as wav format."""
        path = self.get_dst("test.wav")
        s = StreamWriter(path, format="wav")
        s.set_metadata(metadata={"artist": "torchaudio", "title": self.id()})
        s.add_audio_stream(sample_rate, num_channels, format=src_fmt)

        chunk = get_audio_chunk(src_fmt, sample_rate, num_channels)
        with s.open():
            s.write_audio_chunk(0, chunk)

    @parameterized.expand(
        [
            ("mp3", 8000, 1, None, "s32p", None),
            ("mp3", 16000, 2, None, "fltp", None),
            ("mp3", 44100, 1, None, "s16p", {"abr": "true"}),
            ("flac", 8000, 1, None, "s16", None),
            ("flac", 16000, 2, None, "s32", None),
            ("opus", 48000, 2, "opus", None, None),
            ("ogg", 48000, 2, "vorbis", None, None),
            ("adts", 8000, 1, None, "fltp", None),  # AAC format
        ]
    )
    def test_valid_audio_muxer_and_codecs(
        self, ext, sample_rate, num_channels, encoder, encoder_format, encoder_option
    ):
        """Tensor of various dtypes can be saved as given format."""
        path = self.get_dst(f"test.{ext}")
        s = StreamWriter(path, format=ext)
        s.set_metadata(metadata={"artist": "torchaudio", "title": self.id()})
        s.add_audio_stream(
            sample_rate, num_channels, encoder=encoder, encoder_option=encoder_option, encoder_format=encoder_format
        )

        chunk = get_audio_chunk("flt", sample_rate, num_channels)
        with s.open():
            s.write_audio_chunk(0, chunk)

    @nested_params(
        [
            "gray8",
            "rgb24",
            "bgr24",
            "yuv444p",
        ],
        [(128, 64), (720, 576)],
    )
    def test_valid_video_muxer_and_codecs(self, src_format, size):
        """Image tensors of various formats can be saved as mp4"""
        ext = "mp4"
        frame_rate = 10
        width, height = size

        path = self.get_dst(f"test.{ext}")
        s = StreamWriter(path, format=ext)
        s.add_video_stream(frame_rate, width, height, format=src_format)

        chunk = get_video_chunk(src_format, frame_rate, width=width, height=height)
        with s.open():
            s.write_video_chunk(0, chunk)

    def test_valid_audio_video_muxer(self):
        """Audio/image tensors are saved as single video"""
        ext = "mp4"

        sample_rate = 16000
        num_channels = 3

        frame_rate = 30000 / 1001
        width, height = 720, 576
        video_fmt = "yuv444p"

        path = self.get_dst(f"test.{ext}")
        s = StreamWriter(path, format=ext)
        s.set_metadata({"artist": "torchaudio", "title": self.id()})
        s.add_audio_stream(sample_rate, num_channels)
        s.add_video_stream(frame_rate, width, height, format=video_fmt)

        audio = get_audio_chunk("flt", sample_rate, num_channels)
        video = get_video_chunk(video_fmt, frame_rate, height=height, width=width)

        with s.open():
            s.write_audio_chunk(0, audio)
            s.write_video_chunk(1, video)

    @nested_params(
        [
            ("gray8", "gray8"),
            ("rgb24", "rgb24"),
            ("bgr24", "bgr24"),
            ("yuv444p", "yuv444p"),
            ("rgb24", "yuv444p"),
            ("bgr24", "yuv444p"),
        ],
    )
    def test_video_raw_out(self, formats):
        """Verify that viedo out is correct with/without color space conversion"""
        filename = "test.rawvideo"
        frame_rate = 30000 / 1001

        width, height = 720, 576
        src_fmt, encoder_fmt = formats
        frames = int(frame_rate * 2)
        channels = 1 if src_fmt == "gray8" else 3

        # Generate data
        src_size = (frames, channels, height, width)
        chunk = torch.randint(low=0, high=255, size=src_size, dtype=torch.uint8)

        # Write data
        dst = self.get_dst(filename)
        s = StreamWriter(dst, format="rawvideo")
        s.add_video_stream(frame_rate, width, height, format=src_fmt, encoder_format=encoder_fmt)
        with s.open():
            s.write_video_chunk(0, chunk)

        # Fetch the written data
        if self.test_fileobj:
            dst.flush()
        buf = self.get_buf(filename)
        result = torch.frombuffer(buf, dtype=torch.uint8)
        if encoder_fmt.endswith("p"):
            result = result.reshape(src_size)
        else:
            result = result.reshape(frames, height, width, channels).permute(0, 3, 1, 2)

        # check that they are same
        if src_fmt == encoder_fmt:
            expected = chunk
        else:
            if src_fmt == "bgr24":
                chunk = chunk[:, [2, 1, 0], :, :]
            expected = rgb_to_yuv_ccir(chunk)
        self.assertEqual(expected, result, atol=1, rtol=0)

    @nested_params([25, 30], [(78, 96), (240, 426), (360, 640)], ["yuv444p", "rgb24"])
    def test_video_num_frames(self, framerate, resolution, format):
        """Saving video as MP4 properly keep all the frames"""

        ext = "mp4"
        filename = f"test.{ext}"
        h, w = resolution

        # Write data
        dst = self.get_dst(filename)
        s = torchaudio.io.StreamWriter(dst=dst, format=ext)
        s.add_video_stream(frame_rate=framerate, height=h, width=w, format=format)
        chunk = torch.stack([torch.full((3, h, w), i, dtype=torch.uint8) for i in torch.linspace(0, 255, 256)])
        with s.open():
            s.write_video_chunk(0, chunk)
        if self.test_fileobj:
            dst.flush()

        # Load data
        s = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        print(s.get_src_stream_info(0))
        s.add_video_stream(-1)
        s.process_all_packets()
        (saved,) = s.pop_chunks()

        assert saved.shape == chunk.shape

        if format == "yuv444p":
            # The following works if encoder_format is also yuv444p.
            # Otherwise, the typical encoder format is yuv420p which incurs some data loss,
            # and assertEqual fails.
            #
            # This is the case for libx264 encoder, but it's not always available.
            # ffmpeg==4.2 from conda-forge (osx-arm64) comes with it but ffmpeg==5.1.2 does not.
            # Since we do not have function to check the runtime availability of encoders,
            # commenting it out for now.

            # self.assertEqual(saved, chunk)
            pass

    @nested_params(
        ["wav", "flac"],
        [8000, 16000, 44100],
        [1, 2],
    )
    def test_audio_num_frames_lossless(self, ext, sample_rate, num_channels):
        """Lossless format preserves the data"""
        filename = f"test.{ext}"

        data = get_sinusoid(sample_rate=sample_rate, n_channels=num_channels, dtype="int16", channels_first=False)

        # Write data
        dst = self.get_dst(filename)
        s = torchaudio.io.StreamWriter(dst=dst, format=ext)
        s.add_audio_stream(sample_rate=sample_rate, num_channels=num_channels, format="s16")
        with s.open():
            s.write_audio_chunk(0, data)

        if self.test_fileobj:
            dst.flush()

        # Load data
        s = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        s.add_audio_stream(-1)
        s.process_all_packets()
        (saved,) = s.pop_chunks()

        self.assertEqual(saved, data)

    @parameterized.expand(
        [
            ("mp3", 1, 8000),
            ("mp3", 1, 16000),
            ("mp3", 1, 44100),
            ("mp3", 2, 8000),
            ("mp3", 2, 16000),
            ("mp3", 2, 44100),
            ("opus", 1, 48000),
        ]
    )
    def test_audio_num_frames_lossy(self, ext, num_channels, sample_rate):
        """Saving audio preserves the number of channels and frames"""
        filename = f"test.{ext}"

        data = get_sinusoid(sample_rate=sample_rate, n_channels=num_channels, channels_first=False)

        # Write data
        dst = self.get_dst(filename)
        s = torchaudio.io.StreamWriter(dst=dst, format=ext)
        s.add_audio_stream(sample_rate=sample_rate, num_channels=num_channels)
        with s.open():
            s.write_audio_chunk(0, data)

        if self.test_fileobj:
            dst.flush()

        # Load data
        s = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        s.add_audio_stream(-1)
        s.process_all_packets()
        (saved,) = s.pop_chunks()

        # This test fails for OPUS if FFmpeg is 4.1, but it passes for 4.2+
        # 4.1 produces 48312 samples (extra 312)
        # Probably this commit fixes it.
        # https://github.com/FFmpeg/FFmpeg/commit/18aea7bdd96b320a40573bccabea56afeccdd91c
        # TODO: issue warning if 4.1?
        if ext == "opus":
            ver = torchaudio.utils.ffmpeg_utils.get_versions()["libavcodec"]
            # 5.1 libavcodec     59. 18.100
            # 4.4 libavcodec     58.134.100
            # 4.3 libavcodec     58. 91.100
            # 4.2 libavcodec     58. 54.100
            # 4.1 libavcodec     58. 35.100
            if ver[0] < 59 and ver[1] < 54:
                return
        self.assertEqual(saved.shape, data.shape)

    def test_preserve_fps(self):
        """Decimal point frame rate is properly saved

        https://github.com/pytorch/audio/issues/2830
        """
        ext = "mp4"
        filename = f"test.{ext}"
        frame_rate = 5000 / 167
        width, height = 96, 128

        # Write data
        dst = self.get_dst(filename)
        writer = torchaudio.io.StreamWriter(dst=dst, format=ext)
        writer.add_video_stream(frame_rate=frame_rate, width=width, height=height)

        video = torch.randint(256, (90, 3, height, width), dtype=torch.uint8)
        with writer.open():
            writer.write_video_chunk(0, video)
        if self.test_fileobj:
            dst.flush()

        # Load data
        reader = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        assert reader.get_src_stream_info(0).frame_rate == frame_rate

    def test_video_pts_increment(self):
        """PTS values increment by the inverse of frame rate"""

        ext = "mp4"
        num_frames = 256
        filename = f"test.{ext}"
        frame_rate = 5000 / 167
        width, height = 96, 128

        # Write data
        dst = self.get_dst(filename)
        writer = torchaudio.io.StreamWriter(dst=dst, format=ext)
        writer.add_video_stream(frame_rate=frame_rate, width=width, height=height)

        video = torch.randint(256, (num_frames, 3, height, width), dtype=torch.uint8)
        with writer.open():
            writer.write_video_chunk(0, video)

        if self.test_fileobj:
            dst.flush()

        reader = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        reader.add_video_stream(1)
        pts = [chunk.pts for (chunk,) in reader.stream()]
        assert len(pts) == num_frames

        for i, val in enumerate(pts):
            expected = i / frame_rate
            assert abs(val - expected) < 1e-10

    def test_audio_pts_increment(self):
        """PTS values increment by the inverse of sample rate"""

        ext = "wav"
        filename = f"test.{ext}"
        sample_rate = 8000
        num_channels = 2

        # Write data
        dst = self.get_dst(filename)
        writer = torchaudio.io.StreamWriter(dst=dst, format=ext)
        writer.add_audio_stream(sample_rate=sample_rate, num_channels=num_channels)

        audio = get_sinusoid(sample_rate=sample_rate, n_channels=num_channels, channels_first=False)
        num_frames = audio.size(0)
        with writer.open():
            writer.write_audio_chunk(0, audio)

        if self.test_fileobj:
            dst.flush()

        reader = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        frames_per_chunk = sample_rate // 4
        reader.add_audio_stream(frames_per_chunk, -1)

        chunks = [chunk for (chunk,) in reader.stream()]
        expected = num_frames // (frames_per_chunk)
        assert len(chunks) == expected, f"Expected {expected} elements. Found {len(chunks)}"

        num_samples = 0
        for chunk in chunks:
            expected = num_samples / sample_rate
            num_samples += chunk.size(0)
            print(chunk.pts, expected)
            assert abs(chunk.pts - expected) < 1e-10

    @parameterized.expand(
        [
            (10, 100),
            (15, 150),
            (24, 240),
            (25, 200),
            (30, 300),
            (50, 500),
            (60, 600),
            # PTS value conversion involves float <-> int conversion, which can
            # introduce rounding error.
            # This test is a spot-check for popular 29.97 Hz
            (30000 / 1001, 10010),
        ]
    )
    def test_video_pts_overwrite(self, frame_rate, num_frames):
        """Can overwrite PTS"""

        ext = "mp4"
        filename = f"test.{ext}"
        width, height = 8, 8

        # Write data
        dst = self.get_dst(filename)
        writer = torchaudio.io.StreamWriter(dst=dst, format=ext)
        writer.add_video_stream(frame_rate=frame_rate, width=width, height=height)

        video = torch.zeros((1, 3, height, width), dtype=torch.uint8)
        reference_pts = []
        with writer.open():
            for i in range(num_frames):
                pts = i / frame_rate
                reference_pts.append(pts)
                writer.write_video_chunk(0, video, pts)

        # check
        if self.test_fileobj:
            dst.flush()

        reader = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        reader.add_video_stream(1)
        pts = [chunk.pts for (chunk,) in reader.stream()]
        assert len(pts) == len(reference_pts)

        for val, ref in zip(pts, reference_pts):
            # torch provides isclose, but we don't know if converting floats to tensor
            # could introduce a descrepancy, so we compare floats and use math.isclose
            # for that.
            assert math.isclose(val, ref)

    def test_codec_config(self):
        """Can successfully set configuration and write audio."""
        ext = "mp3"
        filename = f"test.{ext}"
        sample_rate = 44100
        num_channels = 2

        # Write data
        dst = self.get_dst(filename)
        writer = torchaudio.io.StreamWriter(dst=dst, format=ext)
        codec_config = torchaudio.io.StreamWriter.CodecConfig(bit_rate=198_000, compression_level=3)
        writer.add_audio_stream(sample_rate=sample_rate, num_channels=num_channels, codec_config=codec_config)

        audio = torch.zeros((8000, 2))
        with writer.open():
            writer.write_audio_chunk(0, audio)

    def test_codec_config_bit_rate_output(self):
        """Increasing the specified bit rate yields a larger encoded output."""
        ext = "mp3"
        sample_rate = 44100
        num_channels = 2
        audio = torch.rand((8000, num_channels))

        def write_audio(buffer, bit_rate):
            writer = torchaudio.io.StreamWriter(dst=buffer, format=ext)
            writer.add_audio_stream(
                sample_rate=sample_rate,
                num_channels=num_channels,
                codec_config=torchaudio.io.StreamWriter.CodecConfig(bit_rate=bit_rate),
            )

            with writer.open():
                writer.write_audio_chunk(0, audio)

        dst = io.BytesIO()
        write_audio(dst, 198_000)
        out0_size = dst.tell()

        dst = io.BytesIO()
        write_audio(dst, 320_000)
        out1_size = dst.tell()

        self.assertGreater(out1_size, out0_size)

    def test_filter_graph_audio(self):
        """Can apply additional effect with filter graph"""
        sample_rate = 8000
        num_channels = 2
        ext = "wav"
        filename = f"test.{ext}"

        original = get_audio_chunk("s16", num_channels=num_channels, sample_rate=sample_rate)

        dst = self.get_dst(filename)
        w = StreamWriter(dst, format=ext)
        w.add_audio_stream(sample_rate=8000, num_channels=num_channels, filter_desc="areverse", format="s16")

        with w.open():
            w.write_audio_chunk(0, original)

        # check
        if self.test_fileobj:
            dst.flush()

        reader = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        reader.add_audio_stream(-1)
        reader.process_all_packets()
        (output,) = reader.pop_chunks()

        self.assertEqual(output, original.flip(0))

    def test_filter_graph_video(self):
        """Can apply additional effect with filter graph"""
        rate = 30
        num_frames, width, height = 400, 160, 90
        ext = "mp4"
        filename = f"test.{ext}"

        original = torch.zeros((num_frames, 3, height, width), dtype=torch.uint8)

        dst = self.get_dst(filename)
        w = StreamWriter(dst, format=ext)
        w.add_video_stream(frame_rate=rate, format="rgb24", height=height, width=width, filter_desc="framestep=2")

        with w.open():
            w.write_video_chunk(0, original)

        # check
        if self.test_fileobj:
            dst.flush()

        reader = torchaudio.io.StreamReader(src=self.get_temp_path(filename))
        reader.add_video_stream(-1)
        reader.process_all_packets()
        (output,) = reader.pop_chunks()

        self.assertEqual(output.shape, [num_frames // 2, 3, height, width])
