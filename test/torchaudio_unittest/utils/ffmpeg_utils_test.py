from torchaudio.utils import ffmpeg_utils
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoFFmpeg


@skipIfNoFFmpeg
class TestFFmpegUtils(PytorchTestCase):
    """Smoke test for ffmpeg_utils module"""

    def tearDown(self):
        ffmpeg_utils.set_log_level(8)
        super().tearDown()

    def test_get_log_level(self):
        """`get_log_level` does not exhibit abnormal behavior"""
        for _ in range(10):
            ffmpeg_utils.get_log_level()

    def test_set_log_level(self):
        """`set_log_level` persists log level"""
        for i in range(-100, 100):
            ffmpeg_utils.set_log_level(i)
            assert ffmpeg_utils.get_log_level() == i

    def test_get_version(self):
        """`get_versions` does not crash"""
        versions = ffmpeg_utils.get_versions()
        assert set(versions.keys()) == {"libavutil", "libavcodec", "libavformat", "libavfilter", "libavdevice"}

    def test_available_stuff(self):
        """get_encoders|decoders|muxers|demuxers|devices function does not segfault"""

        ffmpeg_utils.get_demuxers()
        ffmpeg_utils.get_muxers()
        ffmpeg_utils.get_audio_decoders()
        ffmpeg_utils.get_audio_encoders()
        ffmpeg_utils.get_video_decoders()
        ffmpeg_utils.get_video_encoders()
        ffmpeg_utils.get_input_devices()
        ffmpeg_utils.get_output_devices()
        ffmpeg_utils.get_input_protocols()
        ffmpeg_utils.get_output_protocols()
