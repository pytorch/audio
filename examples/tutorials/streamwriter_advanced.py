"""
StreamWriter Advanced Usage
===========================

**Author**: `Moto Hira <moto@meta.com>`__

This tutorial shows how to use :py:class:`torchaudio.io.StreamWriter` to
play audio and video.

"""

######################################################################
#
# .. note::
#
#    This tutorial uses hardware devices, thus it is not portable across
#    different operating systems.
#
#    The tutorial was written and tested on MacBook Pro (M1, 2020).
#

######################################################################
#
# .. note::
#
#    This tutorial requires FFmpeg libraries.
#    Please refer to :ref:`FFmpeg dependency <ffmpeg_dependency>` for
#    the detail.
#

######################################################################
#
# .. warning::
#
#    TorchAudio dynamically loads compatible FFmpeg libraries
#    installed on the system.
#    The types of supported formats (media format, encoder, encoder
#    options etc) depend on the libraries.
#
#    To check the available devices, muxers and encoders, you can use the
#    following commands
#
#    .. code-block:: console
#
#       ffmpeg -muxers
#       ffmpeg -encoders
#       ffmpeg -devices
#       ffmpeg -protocols

######################################################################
#
# Preparation
# -----------

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

from torchaudio.io import StreamWriter

######################################################################
#

from torchaudio.utils import download_asset

AUDIO_PATH = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
VIDEO_PATH = download_asset(
    "tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"
)

######################################################################
#
# Device Availability
# -------------------
#
# ``StreamWriter`` takes advantage of FFmpeg's IO abstraction and
# writes the data to media devices such as speakers and GUI.
#
# To write to devices, provide ``format`` option to the constructor
# of ``StreamWriter``.
#
# Different OS will have different device options and their availabilities
# depend on the actual installation of FFmpeg.
#
# To check which device is available, you can use `ffmpeg -devices`
# command.
#
# "audiotoolbox" (speaker) and "sdl" (video GUI)
# are available.
#
# .. code-block:: console
#
#    $ ffmpeg -devices
#    ...
#    Devices:
#     D. = Demuxing supported
#     .E = Muxing supported
#     --
#      E audiotoolbox    AudioToolbox output device
#     D  avfoundation    AVFoundation input device
#     D  lavfi           Libavfilter virtual input device
#      E opengl          OpenGL output
#      E sdl,sdl2        SDL2 output device
#
# For details about what devices are available on which OS, please check
# the official FFmpeg documentation. https://ffmpeg.org/ffmpeg-devices.html
#

######################################################################
#
# Playing audio
# -------------
#
# By providing ``format="audiotoolbox"`` option, the StreamWriter writes
# data to speaker device.
#

# Prepare sample audio
waveform, sample_rate = torchaudio.load(AUDIO_PATH, channels_first=False, normalize=False)
num_frames, num_channels = waveform.shape

######################################################################
#

# Configure StreamWriter to write to speaker device
s = StreamWriter(dst="-", format="audiotoolbox")
s.add_audio_stream(sample_rate, num_channels, format="s16")

######################################################################
#

# Write audio to the device
with s.open():
    for i in range(0, num_frames, 256):
        s.write_audio_chunk(0, waveform[i : i + 256])

######################################################################
#
# .. note::
#
#    Writing to "audiotoolbox" is blocking operation, but it will not
#    wait for the aduio playback. The device must be kept open while
#    audio is being played.
#
#    The following code will close the device as soon as the audio is
#    written and before the playback is completed.
#    Adding :py:func:`time.sleep` will help keep the device open until
#    the playback is completed.
#
#    .. code-block::
#
#       with s.open():
#           s.write_audio_chunk(0, waveform)
#

######################################################################
#
# Playing Video
# -------------
#
# To play video, you can use ``format="sdl"`` or ``format="opengl"``.
# Again, you need a version of FFmpeg with corresponding integration
# enabled. The available devices can be checked with ``ffmpeg -devices``.
#
# Here, we use SDL device (https://ffmpeg.org/ffmpeg-devices.html#sdl).
#

# note:
#  SDL device does not support specifying frame rate, and it has to
#  match the refresh rate of display.
frame_rate = 120
width, height = 640, 360


######################################################################
#
# For we define a helper function that delegates the video loading to
# a background thread and give chunks

running = True


def video_streamer(path, frames_per_chunk):
    import queue
    import threading

    from torchaudio.io import StreamReader

    q = queue.Queue()

    # Streaming process that runs in background thread
    def _streamer():
        streamer = StreamReader(path)
        streamer.add_basic_video_stream(
            frames_per_chunk, format="rgb24", frame_rate=frame_rate, width=width, height=height
        )
        for (chunk_,) in streamer.stream():
            q.put(chunk_)
            if not running:
                break

    # Start the background thread and fetch chunks
    t = threading.Thread(target=_streamer)
    t.start()
    while running:
        try:
            yield q.get()
        except queue.Empty:
            break
    t.join()


######################################################################
#
# Now we start streaming. Pressing "Q" will stop the video.
#
# .. note::
#
#    `write_video_chunk` call against SDL device blocks until SDL finishes
#    playing the video.

# Set output device to SDL
s = StreamWriter("-", format="sdl")

# Configure video stream (RGB24)
s.add_video_stream(frame_rate, width, height, format="rgb24", encoder_format="rgb24")

# Play the video
with s.open():
    for chunk in video_streamer(VIDEO_PATH, frames_per_chunk=256):
        try:
            s.write_video_chunk(0, chunk)
        except RuntimeError:
            running = False
            break

######################################################################
#
# .. raw:: html
#
#    <video width="490px" controls autoplay loop muted>
#        <source src="https://download.pytorch.org/torchaudio/tutorial-assets/torchaudio-sdl-demo.mp4">
#    </video>
#
# [`code <https://download.pytorch.org/torchaudio/tutorial-assets/sdl.py>`__]
#

######################################################################
#
# Streaming Video
# ---------------
#
# So far, we looked at how to write to hardware devices. There are some
# alternative methods for video streaming.
#
#

######################################################################
#
# RTMP (Real-Time Messaging Protocol)
# -----------------------------------
#
# Using RMTP, you can stream media (video and/or audio) to a single client.
# This does not require a hardware device, but it requires a separate player.
#
# To use RMTP, specify the protocol and route in ``dst`` argument in
# StreamWriter constructor, then pass ``{"listen": "1"}`` option when opening
# the destination.
#
# StreamWriter will listen to the port and wait for a client to request the video.
# The call to ``open`` is blocked until a request is received.
#
# .. code-block::
#
#    s = StreamWriter(dst="rtmp://localhost:1935/live/app", format="flv")
#    s.add_audio_stream(sample_rate=sample_rate, num_channels=num_channels, encoder="aac")
#    s.add_video_stream(frame_rate=frame_rate, width=width, height=height)
#
#    with s.open(option={"listen": "1"}):
#        for video_chunk, audio_chunk in generator():
#            s.write_audio_chunk(0, audio_chunk)
#            s.write_video_chunk(1, video_chunk)
#
#
# .. raw:: html
#
#    <video width="490px" controls autoplay loop muted>
#        <source src="https://download.pytorch.org/torchaudio/tutorial-assets/torchaudio-rtmp-demo.mp4">
#    </video>
#
# [`code <https://download.pytorch.org/torchaudio/tutorial-assets/rtmp.py>`__]
#


######################################################################
#
# UDP (User Datagram Protocol)
# ----------------------------
#
# Using UDP, you can stream media (video and/or audio) to socket.
# This does not require a hardware device, but it requires a separate player.
#
# Unlike RTMP streaming and client processes are disconnected.
# The streaming process are not aware of client process.
#
# .. code-block::
#
#    s = StreamWriter(dst="udp://localhost:48550", format="mpegts")
#    s.add_audio_stream(sample_rate=sample_rate, num_channels=num_channels, encoder="aac")
#    s.add_video_stream(frame_rate=frame_rate, width=width, height=height)
#
#    with s.open():
#        for video_chunk, audio_chunk in generator():
#            s.write_audio_chunk(0, audio_chunk)
#            s.write_video_chunk(1, video_chunk)
#
# .. raw:: html
#
#    <video width="490px" controls autoplay loop muted>
#        <source src="https://download.pytorch.org/torchaudio/tutorial-assets/torchaudio-udp-demo.mp4">
#    </video>
#
# [`code <https://download.pytorch.org/torchaudio/tutorial-assets/udp.py>`__]
#

######################################################################
#
# Tag: :obj:`torchaudio.io`
