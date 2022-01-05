import pytest
from torchaudio.prototype.ffmpeg import Streamer
from torchaudio.prototype.ffmpeg.io import SourceVideoStream, SourceAudioStream

from .streaming_test_helper import CONFIGS


@pytest.mark.parametrize("src,expected", CONFIGS)
def test_m3u8(src, expected):
    """Test that Streamer can query MPEGTS(M3U8) format"""
    s = Streamer(src)
    assert expected
    sinfo = [s.get_src_stream_info(i) for i in range(s.num_src_streams)]
    assert sinfo == expected
