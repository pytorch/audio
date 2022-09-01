import torch
import torchaudio

from parameterized import parameterized, parameterized_class
from torchaudio_unittest.common_utils import (
    get_asset_path,
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


# TODO:
# Get rid of StreamReader and use synthetic data.


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
            ("mp3", 8000, 1, "s32p", None),
            ("mp3", 16000, 2, "fltp", None),
            ("mp3", 44100, 1, "s16p", {"abr": "true"}),
            ("flac", 8000, 1, "s16", None),
            ("flac", 16000, 2, "s32", None),
            ("opus", 48000, 2, None, {"strict": "experimental"}),
            ("adts", 8000, 1, "fltp", None),  # AAC format
        ]
    )
    def test_valid_audio_muxer_and_codecs(self, ext, sample_rate, num_channels, encoder_format, encoder_option):
        """Tensor of various dtypes can be saved as given format."""
        path = self.get_dst(f"test.{ext}")
        s = StreamWriter(path, format=ext)
        s.set_metadata(metadata={"artist": "torchaudio", "title": self.id()})
        s.add_audio_stream(sample_rate, num_channels, encoder_option=encoder_option, encoder_format=encoder_format)

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
