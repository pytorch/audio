import torchaudio


# If FFmpeg is 4.1 or older
# Tests that checks the number of output samples from OPUS fails
# They work on 4.2+
# Probably this commit fixed it.
# https://github.com/FFmpeg/FFmpeg/commit/18aea7bdd96b320a40573bccabea56afeccdd91c
def lt42():
    ver = torchaudio.utils.ffmpeg_utils.get_versions()["libavcodec"]
    # 5.1 libavcodec     59. 18.100
    # 4.4 libavcodec     58.134.100
    # 4.3 libavcodec     58. 91.100
    # 4.2 libavcodec     58. 54.100
    # 4.1 libavcodec     58. 35.100
    return ver[0] < 59 and ver[1] < 54
