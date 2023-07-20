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


def get_data(height, width, format="yuv444p", frame_rate=30000 / 1001, duration=4):
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
# Benchmark NVENC with StreamWriter
# ---------------------------------
#
# The following function encode the given frames and measure the time
# it takes to encode and the size of the resulting video data.
#

def test_encode(data, encoder, width, height, hw_accel=None, **config):
    buffer = io.BytesIO()
    s = StreamWriter(buffer, format="mp4")
    s.add_video_stream(
        encoder=encoder, width=width, height=height, hw_accel=hw_accel, **config)
    with s.open():
        t0 = time.monotonic()
        s.write_video_chunk(0, data)
        elapsed = time.monotonic() - t0
    size = buffer.tell()
    buffer.seek(0)
    fps = len(data) / elapsed
    print(f" - Processed {len(data)} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")
    print(f" - Encoded data size: {size} bytes")
    return elapsed, size, buffer.read()

######################################################################
#
# We conduct the tests for the following configurations
#
# - Software encoder with the number of threads 1, 4, 8
# - Hardware encoder with and without ``hw_accel`` option.
#

def run_tests(height, width, duration=4):
    print(f"Testing resolution: {width}x{height}")
    pict_config = {
        "height": height,
        "width": width,
        "frame_rate": 30000/1001,
        "format": "yuv444p",
    }

    data = get_data(**pict_config, duration=duration)

    sw_encoder_config = {
        "encoder": "libx264",
        "encoder_format": "yuv444p",
    }

    times = []
    for num_threads in [1, 4, 8]:
        print(f"* Software Encoder (num_threads={num_threads})")
        time_cpu, size_cpu, _ = test_encode(
            data,
            encoder_option={"threads": str(num_threads)},
            **pict_config,
            **sw_encoder_config,
        )
        times.append(time_cpu)

    print(f"* Hardware Encoder")
    encoder_config = {
        "encoder": "h264_nvenc",
        "encoder_format": "yuv444p",
        "encoder_option": {"gpu": "0"}, 
    }

    time_cuda, size_cuda, video_cuda = test_encode(
        data,
        **pict_config,
        **encoder_config,
    )
    times.append(time_cuda)

    print(f"* Hardware Encoder (CUDA frames)")
    time_cuda_accel, _, _ = test_encode(
        data.to(torch.device("cuda:0")),
        **pict_config,
        **encoder_config,
        hw_accel="cuda:0",
    )
    times.append(time_cuda_accel)
    sizes = [size_cpu, size_cuda]
    return times, sizes, video_cuda


######################################################################
#
# And we change the resolution of videos to see how these measurement
# change.
#
# 360P
# ----
#

time_360, size_360, sample_360 = run_tests(360, 640)

######################################################################
# 720P
# ----
#

time_720, size_720, sample_720 = run_tests(720, 1280)

######################################################################
# 1080P
# -----
#

time_1080, size_1080, sample_1080 = run_tests(1080, 1920)

######################################################################
#
# Now we plot the result.
#

def plot():
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=[9.6, 7.2])

    for items in zip(time_360, time_720, time_1080, 'ov^X+'):
        axes[0].plot(items[:-1], marker=items[-1])
    axes[0].grid(axis="both")
    axes[0].set_xticks([0, 1, 2], ["360p", "720p", "1080p"], visible=True)
    axes[0].tick_params(labeltop=False)
    axes[0].legend([
        "Software Encoding (threads=1)",
        "Software Encoding (threads=4)",
        "Software Encoding (threads=8)",
        "Hardware Encoding (CPU Tensor)",
        "Hardware Encoding (CUDA Tensor)",
    ])
    axes[0].set_title("Time to encode videos with different resolutions")
    axes[0].set_ylabel("Time [s]")

    size_cpu, size_cuda = list(zip(size_360, size_720, size_1080))
    axes[1].bar([-0.15, 0.85, 1.85], size_cpu, 0.3, label='Software encoding', edgecolor='black', facecolor='white', hatch='..')
    axes[1].bar([0.15, 1.15, 2.15], size_cuda, 0.3, label='Hardware encoding', edgecolor='black', facecolor='white', hatch='oo')
    axes[1].grid(axis="both")
    axes[1].set_axisbelow(True)
    axes[1].set_xticks([0, 1, 2], ["360p", "720p", "1080p"])
    axes[1].set_ylabel("The encoded size [bytes]")
    axes[1].set_title("The size of encoded videos")
    axes[1].legend()

    plt.tight_layout()
    return fig

plot()

######################################################################
#
# We can see that the time it takes to encode video grows as the
# resolution gets bigger.
#
# In the case of software encoding, increasing the number of threads
# helps reduce the decoding time.
#
# Hardware encoding is faster than software encoding in general.
# Using ``hw_accel`` does not improve the speed of encoding itself
# as much, but if the data are generated by model in CUDA.
#
# The time it takes to move data from CUDA to CPU tensor might add
# up, so it is recommended to test your use case.
#
# The relationship between the resollution of video and the size of
# encoded video file and is a bit tricky to interpret.
#
# Hardware encoder are said to produce larger video size.
# We can see that's the case for 360P. However, this is
# not applicable to 720P and 1080P.
#
# Software encoders are more configurable, and said to produce more
# optimized results. On the other hand, hardware encoders are designed
# for processing speed.

Video(sample_1080, embed=True, width=540, mimetype="video/mp4")

######################################################################
#
# Tag: :obj:`torchaudio.io`
