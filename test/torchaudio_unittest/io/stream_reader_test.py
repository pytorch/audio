import io

import torch
import torchaudio
from unittest import skipIf
from parameterized import parameterized, parameterized_class
from torchaudio_unittest.common_utils import (
    get_asset_path,
    get_image,
    get_sinusoid,
    get_wav_data,
    is_ffmpeg_available,
    nested_params,
    rgb_to_gray,
    rgb_to_yuv_ccir,
    save_image,
    save_wav,
    skipIfNoFFmpeg,
    skipIfNoHWAccel,
    TempDirMixin,
    TorchaudioTestCase,
)


if is_ffmpeg_available():
    from torchaudio.io import StreamReader, StreamWriter
    from torchaudio.io._stream_reader import (
        ChunkTensor,
        OutputAudioStream,
        OutputVideoStream,
        SourceAudioStream,
        SourceStream,
        SourceVideoStream,
    )


@skipIfNoFFmpeg
class ChunkTensorTest(TorchaudioTestCase):
    def test_chunktensor(self):
        """ChunkTensor serves as a replacement of tensor"""
        data = torch.randn((256, 2))
        pts = 16.0

        c = ChunkTensor(data, pts)
        assert c.pts == pts
        self.assertEqual(c, data)

        # method
        sum_ = c.sum()
        assert isinstance(sum_, torch.Tensor)
        self.assertEqual(sum_, data.sum())

        # function form
        min_ = torch.min(c)
        assert isinstance(min_, torch.Tensor)
        self.assertEqual(min_, torch.min(data))

        # attribute
        t = c.T
        assert isinstance(t, torch.Tensor)
        self.assertEqual(t, data.T)

        # in-place op
        c[0] = 0
        self.assertEqual(c, data)

        # pass to other C++ code
        buffer = io.BytesIO()
        w = StreamWriter(buffer, format="wav")
        w.add_audio_stream(8000, 2)
        with w.open():
            w.write_audio_chunk(0, c)
            w.write_audio_chunk(0, c, c.pts)


################################################################################
# Helper decorator and Mixin to duplicate the tests for fileobj
_media_source = parameterized_class(
    ("test_type",),
    [("str",), ("fileobj",)],
    class_name_func=lambda cls, _, params: f'{cls.__name__}_{params["test_type"]}',
)


class _MediaSourceMixin:
    def setUp(self):
        super().setUp()
        self.src = None

    def get_src(self, path):
        if self.src is not None:
            raise ValueError("get_src can be called only once.")

        if self.test_type == "str":
            self.src = path
        elif self.test_type == "fileobj":
            self.src = open(path, "rb")
        return self.src

    def tearDown(self):
        if self.test_type == "fileobj" and self.src is not None:
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
        assert all(k in str(ctx.exception) for k in invalid_keys)

    def test_src_info(self):
        """`get_src_stream_info` properly fetches information"""
        s = StreamReader(self.get_src())
        assert s.num_src_streams == 6

        # Note:
        # Starting from FFmpeg 4.4, audio/video stream metadata
        # include "vendor_id"
        ver = torchaudio.utils.ffmpeg_utils.get_versions()["libavutil"]
        print(ver)
        major, minor, _ = ver
        if major >= 57 or (major == 56 and minor >= 70):
            base_metadata = {"vendor_id": "[0][0][0][0]"}
        else:
            base_metadata = {}

        expected = [
            SourceVideoStream(
                media_type="video",
                codec="h264",
                codec_long_name="H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                format="yuv420p",
                bit_rate=71925,
                num_frames=325,
                bits_per_sample=8,
                metadata=dict(
                    base_metadata,
                    handler_name="\x1fMainconcept Video Media Handler",
                    language="eng",
                ),
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
                num_frames=103,
                bits_per_sample=0,
                metadata=dict(
                    base_metadata,
                    handler_name="#Mainconcept MP4 Sound Media Handler",
                    language="eng",
                ),
                sample_rate=8000.0,
                num_channels=2,
            ),
            SourceStream(
                media_type="subtitle",
                codec="mov_text",
                codec_long_name="MOV text",
                format=None,
                bit_rate=None,
                num_frames=None,
                bits_per_sample=None,
                metadata={
                    "handler_name": "SubtitleHandler",
                    "language": "eng",
                },
            ),
            SourceVideoStream(
                media_type="video",
                codec="h264",
                codec_long_name="H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10",
                format="yuv420p",
                bit_rate=128783,
                num_frames=390,
                bits_per_sample=8,
                metadata=dict(
                    base_metadata,
                    handler_name="\x1fMainconcept Video Media Handler",
                    language="eng",
                ),
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
                num_frames=205,
                bits_per_sample=0,
                metadata=dict(
                    base_metadata,
                    handler_name="#Mainconcept MP4 Sound Media Handler",
                    language="eng",
                ),
                sample_rate=16000.0,
                num_channels=2,
            ),
            SourceStream(
                media_type="subtitle",
                codec="mov_text",
                codec_long_name="MOV text",
                format=None,
                bit_rate=None,
                num_frames=None,
                bits_per_sample=None,
                metadata={
                    "handler_name": "SubtitleHandler",
                    "language": "eng",
                },
            ),
        ]
        output = [s.get_src_stream_info(i) for i in range(6)]
        assert expected == output

    def test_output_info(self):
        s = StreamReader(self.get_src())

        s.add_audio_stream(-1)
        s.add_audio_stream(-1, filter_desc="aresample=8000")
        s.add_audio_stream(-1, filter_desc="aformat=sample_fmts=s16p")
        s.add_video_stream(-1)
        s.add_video_stream(-1, filter_desc="fps=10")
        s.add_video_stream(-1, filter_desc="format=rgb24")
        s.add_video_stream(-1, filter_desc="scale=w=160:h=90")
        expected = [
            OutputAudioStream(
                source_index=4,
                filter_description="anull",
                media_type="audio",
                format="fltp",
                sample_rate=16000.0,
                num_channels=2,
            ),
            OutputAudioStream(
                source_index=4,
                filter_description="aresample=8000",
                media_type="audio",
                format="fltp",
                sample_rate=8000.0,
                num_channels=2,
            ),
            OutputAudioStream(
                source_index=4,
                filter_description="aformat=sample_fmts=s16p",
                media_type="audio",
                format="s16p",
                sample_rate=16000.0,
                num_channels=2,
            ),
            OutputVideoStream(
                source_index=3,
                filter_description="null",
                media_type="video",
                format="yuv420p",
                width=480,
                height=270,
                frame_rate=30000 / 1001,
            ),
            OutputVideoStream(
                source_index=3,
                filter_description="fps=10",
                media_type="video",
                format="yuv420p",
                width=480,
                height=270,
                frame_rate=10,
            ),
            OutputVideoStream(
                source_index=3,
                filter_description="format=rgb24",
                media_type="video",
                format="rgb24",
                width=480,
                height=270,
                frame_rate=30000 / 1001,
            ),
            OutputVideoStream(
                source_index=3,
                filter_description="scale=w=160:h=90",
                media_type="video",
                format="yuv420p",
                width=160,
                height=90,
                frame_rate=30000 / 1001,
            ),
        ]
        output = [s.get_out_stream_info(i) for i in range(s.num_out_streams)]
        assert expected == output

    def test_id3tag(self):
        """get_metadata method can fetch id3tag properly"""
        s = StreamReader(self.get_src("steam-train-whistle-daniel_simon.mp3"))
        output = s.get_metadata()

        expected = {
            "title": "SoundBible.com Must Credit",
            "artist": "SoundBible.com Must Credit",
            "date": "2017",
        }
        assert output == expected

    def test_video_metadata(self):
        """get_metadata method can fetch video metadata"""
        s = StreamReader(self.get_src())
        output = s.get_metadata()

        expected = {
            "compatible_brands": "isomiso2avc1mp41",
            "encoder": "Lavf58.76.100",
            "major_brand": "isom",
            "minor_version": "512",
        }
        assert output == expected

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

    @parameterized.expand(["key", "any", "precise"])
    def test_seek(self, mode):
        """Calling `seek` multiple times should not segfault"""
        s = StreamReader(self.get_src())
        for i in range(10):
            s.seek(i, mode)
        for _ in range(0):
            s.seek(0, mode)
        for i in range(10, 0, -1):
            s.seek(i, mode)

    def test_seek_negative(self):
        """Calling `seek` with negative value should raise an exception"""
        s = StreamReader(self.get_src())
        with self.assertRaises(RuntimeError):
            s.seek(-1.0)

    def test_seek_invalid_mode(self):
        """Calling `seek` with an invalid model should raise an exception"""
        s = StreamReader(self.get_src())
        with self.assertRaises(ValueError):
            s.seek(10, "magic_seek")

    @parameterized.expand(
        [
            # Test keyframe seek
            # The source mp4 video has two key frames the first frame and 203rd frame at 8.08 second.
            # If the seek time stamp is smaller than 8.08, it will seek into the first frame at 0.0 second.
            ("nasa_13013.mp4", "key", 0.2, (0, slice(None))),
            ("nasa_13013.mp4", "key", 8.04, (0, slice(None))),
            ("nasa_13013.mp4", "key", 8.08, (0, slice(202, None))),
            ("nasa_13013.mp4", "key", 8.12, (0, slice(202, None))),
            # The source avi video has one keyframe every twelve frames 0, 12, 24,.. or every 0.4004 seconds.
            # if we seek to a time stamp smaller than 0.4004 it will seek into the first frame at 0.0 second.
            ("nasa_13013.avi", "key", 0.2, (0, slice(None))),
            ("nasa_13013.avi", "key", 1.01, (0, slice(24, None))),
            ("nasa_13013.avi", "key", 7.37, (0, slice(216, None))),
            ("nasa_13013.avi", "key", 7.7, (0, slice(216, None))),
            # Test precise seek
            ("nasa_13013.mp4", "precise", 0.0, (0, slice(None))),
            ("nasa_13013.mp4", "precise", 0.2, (0, slice(5, None))),
            ("nasa_13013.mp4", "precise", 8.04, (0, slice(201, None))),
            ("nasa_13013.mp4", "precise", 8.08, (0, slice(202, None))),
            ("nasa_13013.mp4", "precise", 8.12, (0, slice(203, None))),
            ("nasa_13013.avi", "precise", 0.0, (0, slice(None))),
            ("nasa_13013.avi", "precise", 0.2, (0, slice(1, None))),
            ("nasa_13013.avi", "precise", 8.1, (0, slice(238, None))),
            ("nasa_13013.avi", "precise", 8.14, (0, slice(239, None))),
            ("nasa_13013.avi", "precise", 8.17, (0, slice(240, None))),
            # Test precise seek on video with missing PTS
            ("RATRACE_wave_f_nm_np1_fr_goo_37.avi", "precise", 0.0, (0, slice(None))),
            ("RATRACE_wave_f_nm_np1_fr_goo_37.avi", "precise", 0.2, (0, slice(4, None))),
            ("RATRACE_wave_f_nm_np1_fr_goo_37.avi", "precise", 0.3, (0, slice(7, None))),
            # Test any seek
            # The source avi video has one keyframe every twelve frames 0, 12, 24,.. or every 0.4004 seconds.
            ("nasa_13013.avi", "any", 0.0, (0, slice(None))),
            ("nasa_13013.avi", "any", 0.56, (0, slice(12, None))),
            ("nasa_13013.avi", "any", 7.77, (0, slice(228, None))),
            ("nasa_13013.avi", "any", 0.2002, (11, slice(12, None))),
            ("nasa_13013.avi", "any", 0.233567, (10, slice(12, None))),
            ("nasa_13013.avi", "any", 0.266933, (9, slice(12, None))),
        ]
    )
    def test_seek_modes(self, src, mode, seek_time, ref_indices):
        """We expect the following behaviour from the diferent kinds of seek:
            - `key`: the reader will seek to the first keyframe from the timestamp given
            - `precise`: the reader will seek to the first keyframe from the timestamp given
               and start decoding from that position until the given timestmap (discarding all frames in between)
            - `any`: the  reader will seek to the colsest frame to the timestamp
               given but if this is not a keyframe, the content will be the delta from other frames

        To thest this behaviour we can parameterize the test with the tupple ref_indices. ref_indices[0]
        is the expected index on the frames list decoded after seek and ref_indices[1] is exepected index for
        the list of all frames decoded from the begining (reference frames). This test checks if
        the reference frame at index ref_indices[1] is the same as ref_indices[0]. Plese note that with `any`
        and `key` seek we only compare keyframes, but with `precise` seek we can compare any frame content.
        """
        # Using the first video stream (which is not default video stream)
        stream_index = 0
        # Decode all frames for reference
        src_bin = self.get_src(src)
        s = StreamReader(src_bin)
        s.add_basic_video_stream(-1, stream_index=stream_index)
        s.process_all_packets()
        (ref_frames,) = s.pop_chunks()

        s.seek(seek_time, mode=mode)
        s.process_all_packets()
        (frame,) = s.pop_chunks()

        hyp_index, ref_index = ref_indices

        hyp, ref = frame[hyp_index:], ref_frames[ref_index]
        print(hyp.shape, ref.shape)
        self.assertEqual(hyp, ref)

    @parameterized.expand(
        [
            ("nasa_13013.mp4", [195, 3, 270, 480]),
            # RATRACE does not have valid PTS metadata.
            ("RATRACE_wave_f_nm_np1_fr_goo_37.avi", [36, 3, 240, 560]),
        ]
    )
    def test_change_fps(self, src, shape):
        """Can change the FPS of videos"""
        tgt_frame_rate = 15
        s = StreamReader(self.get_src(src))
        info = s.get_src_stream_info(s.default_video_stream)
        assert info.frame_rate != tgt_frame_rate
        s.add_basic_video_stream(frames_per_chunk=-1, frame_rate=tgt_frame_rate)
        s.process_all_packets()
        (chunk,) = s.pop_chunks()

        assert chunk.shape == torch.Size(shape)

    def test_invalid_chunk_option(self):
        """Passing invalid `frames_per_chunk` and `buffer_chunk_size` raises error"""
        s = StreamReader(self.get_src())
        for fpc, bcs in ((0, 3), (3, 0), (-2, 3), (3, -2)):
            with self.assertRaises(RuntimeError):
                s.add_audio_stream(frames_per_chunk=fpc, buffer_chunk_size=bcs)
            with self.assertRaises(RuntimeError):
                s.add_video_stream(frames_per_chunk=fpc, buffer_chunk_size=bcs)

    def test_unchunked_stream(self):
        """`frames_per_chunk=-1` disable chunking.

        When chunking is disabled, frames contained in one AVFrame become one chunk.
        For video, that is always one frame, but for audio, it depends.
        """
        s = StreamReader(self.get_src())
        s.add_video_stream(frames_per_chunk=-1, buffer_chunk_size=10000)
        s.add_audio_stream(frames_per_chunk=-1, buffer_chunk_size=10000)
        s.process_all_packets()
        video, audio = s.pop_chunks()
        assert video.shape == torch.Size([390, 3, 270, 480])
        assert audio.shape == torch.Size([208896, 2])

    @parameterized.expand([(1,), (3,), (5,), (10,)])
    def test_frames_per_chunk(self, fpc):
        """Changing frames_per_chunk does not change the returned content"""
        src = self.get_src()
        s = StreamReader(src)
        s.add_video_stream(frames_per_chunk=-1, buffer_chunk_size=-1)
        s.add_audio_stream(frames_per_chunk=-1, buffer_chunk_size=-1)
        s.process_all_packets()
        ref_video, ref_audio = s.pop_chunks()

        if self.test_type == "fileobj":
            src.seek(0)

        s = StreamReader(src)
        s.add_video_stream(frames_per_chunk=fpc, buffer_chunk_size=-1)
        s.add_audio_stream(frames_per_chunk=fpc, buffer_chunk_size=-1)
        chunks = list(s.stream())
        video_chunks = torch.cat([c[0] for c in chunks if c[0] is not None])
        audio_chunks = torch.cat([c[1] for c in chunks if c[1] is not None])
        self.assertEqual(ref_video, video_chunks)
        self.assertEqual(ref_audio, audio_chunks)

    def test_buffer_chunk_size(self):
        """`buffer_chunk_size=-1` does not drop frames."""
        src = self.get_src()
        s = StreamReader(src)
        s.add_video_stream(frames_per_chunk=30, buffer_chunk_size=-1)
        s.add_audio_stream(frames_per_chunk=16000, buffer_chunk_size=-1)
        s.process_all_packets()
        for _ in range(13):
            video, audio = s.pop_chunks()
            assert video.shape == torch.Size([30, 3, 270, 480])
            assert audio.shape == torch.Size([16000, 2])
        video, audio = s.pop_chunks()
        assert video is None
        assert audio.shape == torch.Size([896, 2])

        if self.test_type == "fileobj":
            src.seek(0)

        s = StreamReader(src)
        s.add_video_stream(frames_per_chunk=30, buffer_chunk_size=3)
        s.add_audio_stream(frames_per_chunk=16000, buffer_chunk_size=3)
        s.process_all_packets()
        for _ in range(2):
            video, audio = s.pop_chunks()
            assert video.shape == torch.Size([30, 3, 270, 480])
            assert audio.shape == torch.Size([16000, 2])
        video, audio = s.pop_chunks()
        assert video.shape == torch.Size([30, 3, 270, 480])
        assert audio.shape == torch.Size([896, 2])

    @parameterized.expand([(1,), (3,), (5,), (10,)])
    def test_video_pts(self, fpc):
        """PTS values of the first frame are reported in .pts attribute"""
        rate, num_frames = 30000 / 1001, 390
        ref_pts = [i / rate for i in range(0, num_frames, fpc)]

        s = StreamReader(self.get_src())
        s.add_video_stream(fpc)
        pts = [video.pts for video, in s.stream()]
        self.assertEqual(pts, ref_pts)

    @parameterized.expand([(256,), (512,), (1024,), (4086,)])
    def test_audio_pts(self, fpc):
        """PTS values of the first frame are reported in .pts attribute"""
        rate, num_frames = 16000, 208896
        ref_pts = [i / rate for i in range(0, num_frames, fpc)]

        s = StreamReader(self.get_src())
        s.add_audio_stream(fpc, buffer_chunk_size=-1)
        pts = [audio.pts for audio, in s.stream()]
        self.assertEqual(pts, ref_pts)

    def test_pts_unchunked_process_all(self):
        """PTS is zero when loading the entire media with unchunked buffer"""
        s = StreamReader(self.get_src())
        s.add_audio_stream(-1, buffer_chunk_size=-1)
        s.add_video_stream(-1, buffer_chunk_size=-1)
        s.process_all_packets()
        audio, video = s.pop_chunks()
        assert audio.pts == 0.0
        assert video.pts == 0.0
        assert audio.size(0) == 208896
        assert video.size(0) == 390

    def test_pts_unchunked(self):
        """PTS grows proportionally to the number of frames decoded"""
        s = StreamReader(self.get_src())
        s.add_audio_stream(-1, buffer_chunk_size=-1)
        s.add_video_stream(-1, buffer_chunk_size=-1)

        num_audio_frames, num_video_frames = 0, 0
        while num_audio_frames < 208896 and num_video_frames < 390:
            s.process_packet()
            audio, video = s.pop_chunks()
            if audio is None and video is None:
                continue
            if audio is not None:
                assert audio.pts == num_audio_frames / 16000
                num_audio_frames += audio.size(0)
            if video is not None:
                assert video.pts == num_video_frames * 1001 / 30000
                num_video_frames += video.size(0)


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
        if self.test_type == "fileobj":
            src.seek(0)
        self._test_wav(src, original, fmt=None)

    def test_audio_stream_format(self):
        "`format` argument properly changes the sample format of decoded audio"
        num_channels = 2
        src, s32 = self.get_src(8000, dtype="int32", num_channels=num_channels)
        args = {
            "num_channels": num_channels,
            "normalize": False,
            "channels_first": False,
            "num_frames": 1 << 16,
        }
        u8 = get_wav_data("uint8", **args)
        s16 = get_wav_data("int16", **args)
        s64 = s32.to(torch.int64) * (1 << 32)
        f32 = get_wav_data("float32", **args)
        f64 = get_wav_data("float64", **args)

        s = StreamReader(src)
        s.add_basic_audio_stream(frames_per_chunk=-1, format="u8")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="u8p")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="s16")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="s16p")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="s32")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="s32p")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="s64")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="s64p")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="flt")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="fltp")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="dbl")
        s.add_basic_audio_stream(frames_per_chunk=-1, format="dblp")
        s.process_all_packets()
        chunks = s.pop_chunks()
        self.assertEqual(chunks[0], u8, atol=1, rtol=0)
        self.assertEqual(chunks[1], u8, atol=1, rtol=0)
        self.assertEqual(chunks[2], s16)
        self.assertEqual(chunks[3], s16)
        self.assertEqual(chunks[4], s32)
        self.assertEqual(chunks[5], s32)
        self.assertEqual(chunks[6], s64)
        self.assertEqual(chunks[7], s64)
        self.assertEqual(chunks[8], f32)
        self.assertEqual(chunks[9], f32)
        self.assertEqual(chunks[10], f64)
        self.assertEqual(chunks[11], f64)

    @nested_params([4000, 16000])
    def test_basic_audio_stream_sample_rate(self, sr):
        """`sample_rate` argument changes the sample_rate of decoded audio"""
        src_num_channels, src_sr = 2, 8000
        data = get_sinusoid(sample_rate=src_sr, n_channels=src_num_channels, channels_first=False)
        path = self.get_temp_path("ref.wav")
        save_wav(path, data, src_sr, channels_first=False)

        s = StreamReader(path)
        s.add_basic_audio_stream(frames_per_chunk=-1, format="flt", sample_rate=sr)
        self.assertEqual(s.get_src_stream_info(0).sample_rate, src_sr)
        self.assertEqual(s.get_out_stream_info(0).sample_rate, sr)

        s.process_all_packets()
        (chunks,) = s.pop_chunks()
        self.assertEqual(chunks.shape, [sr, src_num_channels])

    @nested_params([1, 2, 3, 8, 16])
    def test_basic_audio_stream_num_channels(self, num_channels):
        """`sample_rate` argument changes the number of channels of decoded audio"""
        src_num_channels, sr = 2, 8000
        data = get_sinusoid(sample_rate=sr, n_channels=src_num_channels, channels_first=False)
        path = self.get_temp_path("ref.wav")
        save_wav(path, data, sr, channels_first=False)

        s = StreamReader(path)
        s.add_basic_audio_stream(frames_per_chunk=-1, format="flt", num_channels=num_channels)
        self.assertEqual(s.get_src_stream_info(0).num_channels, src_num_channels)
        self.assertEqual(s.get_out_stream_info(0).num_channels, num_channels)

        s.process_all_packets()
        (chunks,) = s.pop_chunks()
        self.assertEqual(chunks.shape, [sr, num_channels])

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
            if self.test_type == "fileobj":
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

    def test_png_yuv_read_out(self):
        """Providing format prpoerly change the color space"""
        rgb = torch.empty(1, 3, 256, 256, dtype=torch.uint8)
        rgb[0, 0] = torch.arange(256, dtype=torch.uint8).reshape([1, -1])
        rgb[0, 1] = torch.arange(256, dtype=torch.uint8).reshape([-1, 1])
        alpha = torch.full((1, 1, 256, 256), 255, dtype=torch.uint8)
        for i in range(256):
            rgb[0, 2] = i
            path = self.get_temp_path(f"ref_{i}.png")
            save_image(path, rgb[0], mode="RGB")

            rgb16 = ((rgb.to(torch.int32) - 128) << 8).to(torch.int16)

            yuv = rgb_to_yuv_ccir(rgb)
            yuv16 = yuv.to(torch.int16) * 4
            bgr = rgb[:, [2, 1, 0], :, :]
            gray = rgb_to_gray(rgb)
            argb = torch.cat([alpha, rgb], dim=1)
            rgba = torch.cat([rgb, alpha], dim=1)
            abgr = torch.cat([alpha, bgr], dim=1)
            bgra = torch.cat([bgr, alpha], dim=1)

            s = StreamReader(path)
            s.add_basic_video_stream(frames_per_chunk=-1, format="yuv444p")
            s.add_basic_video_stream(frames_per_chunk=-1, format="yuv420p")
            s.add_basic_video_stream(frames_per_chunk=-1, format="nv12")
            s.add_basic_video_stream(frames_per_chunk=-1, format="rgb24")
            s.add_basic_video_stream(frames_per_chunk=-1, format="bgr24")
            s.add_basic_video_stream(frames_per_chunk=-1, format="gray8")
            s.add_basic_video_stream(frames_per_chunk=-1, format="rgb48le")
            s.add_basic_video_stream(frames_per_chunk=-1, format="argb")
            s.add_basic_video_stream(frames_per_chunk=-1, format="rgba")
            s.add_basic_video_stream(frames_per_chunk=-1, format="abgr")
            s.add_basic_video_stream(frames_per_chunk=-1, format="bgra")
            s.add_basic_video_stream(frames_per_chunk=-1, format="yuv420p10le")
            s.process_all_packets()
            chunks = s.pop_chunks()
            self.assertEqual(chunks[0], yuv, atol=1, rtol=0)
            self.assertEqual(chunks[1], yuv, atol=1, rtol=0)
            self.assertEqual(chunks[2], yuv, atol=1, rtol=0)
            self.assertEqual(chunks[3], rgb, atol=0, rtol=0)
            self.assertEqual(chunks[4], bgr, atol=0, rtol=0)
            self.assertEqual(chunks[5], gray, atol=1, rtol=0)
            self.assertEqual(chunks[6], rgb16, atol=256, rtol=0)
            self.assertEqual(chunks[7], argb, atol=0, rtol=0)
            self.assertEqual(chunks[8], rgba, atol=0, rtol=0)
            self.assertEqual(chunks[9], abgr, atol=0, rtol=0)
            self.assertEqual(chunks[10], bgra, atol=0, rtol=0)
            self.assertEqual(chunks[11], yuv16, atol=4, rtol=0)


@skipIfNoHWAccel("h264_cuvid")
class CuvidHWAccelInterfaceTest(TorchaudioTestCase):
    def test_dup_hw_acel(self):
        """Specifying the same source stream with and without HW accel should fail (instead of segfault later)"""
        src = get_asset_path("nasa_13013.mp4")
        r = StreamReader(src)
        r.add_video_stream(-1, decoder="h264_cuvid")
        with self.assertRaises(RuntimeError):
            r.add_video_stream(-1, decoder="h264_cuvid", hw_accel="cuda")

        r = StreamReader(src)
        r.add_video_stream(-1, decoder="h264_cuvid", hw_accel="cuda")
        with self.assertRaises(RuntimeError):
            r.add_video_stream(-1, decoder="h264_cuvid")


@_media_source
class CudaDecoderTest(_MediaSourceMixin, TempDirMixin, TorchaudioTestCase):
    @skipIfNoHWAccel("h264_cuvid")
    def test_h264_cuvid(self):
        """GPU decoder works for H264"""
        src = self.get_src(get_asset_path("nasa_13013.mp4"))
        r = StreamReader(src)
        r.add_video_stream(10, decoder="h264_cuvid")

        num_frames = 0
        for (chunk,) in r.stream():
            self.assertEqual(chunk.device, torch.device("cpu"))
            self.assertEqual(chunk.dtype, torch.uint8)
            self.assertEqual(chunk.shape, torch.Size([10, 3, 270, 480]))
            num_frames += chunk.size(0)
        assert num_frames == 390

    @skipIfNoHWAccel("h264_cuvid")
    def test_h264_cuvid_hw_accel(self):
        """GPU decoder works for H264 with HW acceleration, and put the frames on CUDA tensor"""
        src = self.get_src(get_asset_path("nasa_13013.mp4"))
        r = StreamReader(src)
        r.add_video_stream(10, decoder="h264_cuvid", hw_accel="cuda")

        num_frames = 0
        for (chunk,) in r.stream():
            self.assertEqual(chunk.device, torch.device("cuda:0"))
            self.assertEqual(chunk.dtype, torch.uint8)
            self.assertEqual(chunk.shape, torch.Size([10, 3, 270, 480]))
            num_frames += chunk.size(0)
        assert num_frames == 390

    @skipIfNoHWAccel("hevc_cuvid")
    def test_hevc_cuvid(self):
        """GPU decoder works for H265/HEVC"""
        src = self.get_src(get_asset_path("testsrc.hevc"))
        r = StreamReader(src)
        r.add_video_stream(10, decoder="hevc_cuvid")

        num_frames = 0
        for (chunk,) in r.stream():
            self.assertEqual(chunk.device, torch.device("cpu"))
            self.assertEqual(chunk.dtype, torch.uint8)
            self.assertEqual(chunk.shape, torch.Size([10, 3, 144, 256]))
            num_frames += chunk.size(0)
        assert num_frames == 300

    @skipIfNoHWAccel("hevc_cuvid")
    def test_hevc_cuvid_hw_accel(self):
        """GPU decoder works for H265/HEVC with HW acceleration, and put the frames on CUDA tensor"""
        src = self.get_src(get_asset_path("testsrc.hevc"))
        r = StreamReader(src)
        r.add_video_stream(10, decoder="hevc_cuvid", hw_accel="cuda")

        num_frames = 0
        for (chunk,) in r.stream():
            self.assertEqual(chunk.device, torch.device("cuda:0"))
            self.assertEqual(chunk.dtype, torch.int16)
            self.assertEqual(chunk.shape, torch.Size([10, 3, 144, 256]))
            num_frames += chunk.size(0)
        assert num_frames == 300


@skipIfNoHWAccel("h264_cuvid")
@skipIf(True, "Skip since failing see issue: https://github.com/pytorch/audio/issues/3376")
class FilterGraphWithCudaAccel(TorchaudioTestCase):
    def test_sclae_cuda_change_size(self):
        """scale_cuda filter can be used when HW accel is on"""
        src = get_asset_path("nasa_13013.mp4")
        r = StreamReader(src)
        r.add_video_stream(10, decoder="h264_cuvid", hw_accel="cuda", filter_desc="scale_cuda=iw/2:ih/2")
        num_frames = 0
        for (chunk,) in r.stream():
            self.assertEqual(chunk.device, torch.device("cuda:0"))
            self.assertEqual(chunk.dtype, torch.uint8)
            self.assertEqual(chunk.shape, torch.Size([10, 3, 135, 240]))
            num_frames += chunk.size(0)
        assert num_frames == 390

    def test_scale_cuda_format(self):
        """yuv444p format conversion should work"""
        src = get_asset_path("nasa_13013.mp4")
        r = StreamReader(src)
        r.add_video_stream(10, decoder="h264_cuvid", hw_accel="cuda", filter_desc="scale_cuda=format=yuv444p")
        num_frames = 0
        for (chunk,) in r.stream():
            self.assertEqual(chunk.device, torch.device("cuda:0"))
            self.assertEqual(chunk.dtype, torch.uint8)
            self.assertEqual(chunk.shape, torch.Size([10, 3, 270, 480]))
            num_frames += chunk.size(0)
        assert num_frames == 390
