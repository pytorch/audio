"""
Accelerated video encoding with NVENC
=====================================

.. _nvenc_tutorial:

**Author**: `Moto Hira <moto@meta.com>`__

This tutorial shows how to use NVIDIA’s hardware video encoder (NVENC)
with TorchAudio, and how it improves the performance of video encoding.
"""

######################################################################
#
# .. note::
#
#    This tutorial requires FFmpeg libraries compiled with HW
#    acceleration enabled.
#
#    Please refer to
#    :ref:`Enabling GPU video decoder/encoder <enabling_hw_decoder>`
#    for how to build FFmpeg with HW acceleration.
#
# .. note::
#
#    Most modern GPUs have both HW decoder and encoder, but some
#    highend GPUs like A100 and H100 do not have HW encoder.
#    Please refer to the following for the availability and
#    format coverage.
#    https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
#
#    Attempting to use HW encoder on these GPUs fails with an error
#    message like ``Generic error in an external library``.
#    You can enable debug log with
#    :py:func:`torchaudio.utils.ffmpeg_utils.set_log_level` to see more
#    detailed error messages issued along the way.
#

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

import io
import time

import matplotlib.pyplot as plt
from IPython.display import Video
from torchaudio.io import StreamReader, StreamWriter

######################################################################
#
# Check the prerequisites
# -----------------------
#
# First, we check that TorchAudio correctly detects FFmpeg libraries
# that support HW decoder/encoder.
#

from torchaudio.utils import ffmpeg_utils

######################################################################
#

print("FFmpeg Library versions:")
for k, ver in ffmpeg_utils.get_versions().items():
    print(f"  {k}: {'.'.join(str(v) for v in ver)}")

######################################################################
#
print("Available NVENC Encoders:")
for k in ffmpeg_utils.get_video_encoders().keys():
    if "nvenc" in k:
        print(f" - {k}")

######################################################################
#

print("Avaialbe GPU:")
print(torch.cuda.get_device_properties(0))


######################################################################
#
# We use the following helper function to generate test frame data.
# For the detail of synthetic video generation please refer to
# :ref:`StreamReader Advanced Usage <lavfi>`.


def get_data(height, width, format="yuv444p", frame_rate=30000 / 1001, duration=3):
    src = f"testsrc2=rate={frame_rate}:size={width}x{height}:duration={duration}"
    s = StreamReader(src=src, format="lavfi")
    s.add_basic_video_stream(-1, format=format)
    s.process_all_packets()
    (video,) = s.pop_chunks()
    return video


######################################################################
# Encoding videos with NVENC
# --------------------------
#
# To use HW video encoder, you need to specify the HW encoder when
# defining the output video stream.
#
# To do so, you need to use
# :py:meth:`~torchaudio.io.StreamWriter.add_video_stream`
# method, and provide ``encoder`` option.
#
#

######################################################################
#

pict_config = {
    "height": 360,
    "width": 640,
    "frame_rate": 30000 / 1001,
    "format": "yuv444p",
}

frame_data = get_data(**pict_config)

######################################################################
#

w = StreamWriter(io.BytesIO(), format="mp4")
w.add_video_stream(**pict_config, encoder="h264_nvenc", encoder_format="yuv444p")
with w.open():
    w.write_video_chunk(0, frame_data)

######################################################################
# Similar to the HW decoder, by default, the encoder expects the frame
# data to be on CPU memory. To send data from CUDA memory, you need to
# specify ``hw_accel`` option.
#

buffer = io.BytesIO()
w = StreamWriter(buffer, format="mp4")
w.add_video_stream(**pict_config, encoder="h264_nvenc", encoder_format="yuv444p", hw_accel="cuda:0")
with w.open():
    w.write_video_chunk(0, frame_data.to(torch.device("cuda:0")))
buffer.seek(0)
video_cuda = buffer.read()

Video(video_cuda, embed=True, mimetype="video/mp4")

######################################################################
# .. note::
#
#    It is known that GPU encoders generate bigger files than
#    software encoders.
#

buffer = io.BytesIO()
w = StreamWriter(buffer, format="mp4")
w.add_video_stream(**pict_config, encoder="libx264", encoder_format="yuv444p")
with w.open():
    w.write_video_chunk(0, frame_data)
buffer.seek(0)
video_cpu = buffer.read()

######################################################################
#

print("The size of the video encoded with software encoder: ", len(video_cpu))
print("The size of the video encoded with hardware encoder: ", len(video_cuda))


######################################################################
# Benchmark NVENC with StreamWriter
# ---------------------------------
#

cuda = torch.device("cuda:0")


def test_encode(data, encoder, width, height, hw_accel=None, **config):
    buffer = io.BytesIO()
    s = StreamWriter(buffer, format="mp4")
    s.add_video_stream(
        encoder=encoder, width=width, height=height, hw_accel=hw_accel, **config)
    with s.open():
        t0 = time.monotonic()
        s.write_video_chunk(0, data)
        elapsed = time.monotonic() - t0
    fps = len(data) / elapsed
    size = buffer.tell()
    print(f"Test: {encoder}{'' if hw_accel is None else ' (CUDA frames)'}")
    print(
        f" - Processed {len(data)} frames ({width}x{height}) in {elapsed:.2f} seconds."
        f" ({fps:7.2f} fps) Encoded data size: {size} bytes")
    return fps, size


######################################################################
# 360P
# ----
#

pict_config = {
    "height": 360,
    "width": 640,
    "frame_rate": 30000/1001,
    "format": "yuv444p",
}

data = get_data(**pict_config, duration=30)

######################################################################
#

encoder_config = {
    "encoder": "libx264",
    "encoder_format": "yuv444p",
    "encoder_option": {"threads": "4"},
}

fps_360_cpu, size_360_cpu = test_encode(data, **pict_config, **encoder_config)

######################################################################
#

encoder_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"}, 
}

fps_360_cuda, size_360_cuda = test_encode(data, **pict_config, **encoder_config)

######################################################################
#

encoder_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"},
    "hw_accel": "cuda:0"
}

fps_360_cuda_hw, _ = test_encode(data.to(cuda), **pict_config, **encoder_config)

######################################################################
# 720P
# ----
#

pict_config = {
    "height": 720,
    "width": 1280,
    "frame_rate": 30000/1001,
    "format": "yuv444p",
}

data = get_data(**pict_config, duration=10)

######################################################################
#

encoder_config = {
    "encoder": "libx264",
    "encoder_format": "yuv444p",
    "encoder_option": {"threads": "4"},
}

fps_720_cpu, size_720_cpu = test_encode(data, **pict_config, **encoder_config)

######################################################################
#

encoder_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"}, 
}

fps_720_cuda, size_720_cuda = test_encode(data, **pict_config, **encoder_config)

######################################################################
#

encoder_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"},
    "hw_accel": "cuda:0"
}

fps_720_cuda_hw, _ = test_encode(data.to(cuda), **pict_config, **encoder_config)

######################################################################
# 1080P
# ----
#

pict_config = {
    "height": 1080,
    "width": 1920,
    "frame_rate": 30000/1001,
    "format": "yuv444p",
}

data = get_data(**pict_config, duration=5)

######################################################################
#

encoder_config = {
    "encoder": "libx264",
    "encoder_format": "yuv444p",
    "encoder_option": {"threads": "4"},
}

fps_1080_cpu, size_1080_cpu = test_encode(data, **pict_config, **encoder_config)

######################################################################
#

encoder_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"}, 
}

fps_1080_cuda, size_1080_cuda = test_encode(data, **pict_config, **encoder_config)

######################################################################
#

encoder_config = {
    "encoder": "h264_nvenc",
    "encoder_format": "yuv444p",
    "encoder_option": {"gpu": "0"},
    "hw_accel": "cuda:0"
}

fps_1080_cuda_hw, _ = test_encode(data.to(cuda), **pict_config, **encoder_config)

######################################################################
#

######################################################################
#
# Tag: :obj:`torchaudio.io`
