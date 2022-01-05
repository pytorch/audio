"""
Streaming API Tutorial
======================

This tutorial shows how to use torchaudio's prototype streaming API.

"""

######################################################################
# Overview
# --------
#
# Streaming API leverages the powerful I/O features of ffmpeg.
#
# It can
#  - Load audio/video in variety of formats
#  - Load audio/video from local/remote source
#  - Load audio from microphone device
#  - Load video from video device / screen
#  - Load audio/video chunk by chunk
#  - Change the sample rate / frame rate, image size, on-the-fly
#  - Apply variety of filters and preprocessings
#
# At this moment, the features that the ffmpeg integration provide
# are limited to the form of
#
# `<some media source> -> <optional processing> -> <tensor>`
#
# If you have other forms that can be useful to your usecases,
# please file a feature request.
#

######################################################################
# Preparation
# -----------
#

# !add-apt-repository -y ppa:jonathonf/ffmpeg-4
# !apt install -y pkg-config libavfilter-dev libavdevice-dev
# !pip uninstall -y torchaudio
# !pip install 'cmake>3.18' ninja
# !BUILD_SOX=0 BUILD_KALDI=0 BUILD_CTC_DECODER=0 BUILD_RNNT=0 pip install --verbose --no-use-pep517 git+https://github.com/mthrok/audio.git@export-D33324389

######################################################################
#

import IPython
import matplotlib.pyplot as plt
from torchaudio.prototype.ffmpeg import Streamer

######################################################################
# Dealing with formats with multiple streams
# ------------------------------------------
#
# Some container formats, such as `mp4` and `mkv` may contain
# multiple media streams, such as videos with different resolutions
# and/or audios with different sampling rates.
#
# You can configure the output streams according to your needs.
# In the following, we use an M3U8 playlist, which contains dozens
# of video streams.
#

src = "https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8"

######################################################################
# Here is the preview of the video we stream.

# Note:
# Workaround to display m3u8 as Chrome won't play it directly..
IPython.display.HTML(
    f"""
<link href="https://vjs.zencdn.net/7.17.0/video-js.css" rel="stylesheet">
<script src="https://vjs.zencdn.net/7.17.0/video.min.js"></script>
<video id="sample" width="960" height="540" class="video-js vjs-fluid vjs-default-skin" controls>
  <source src="{src}" type="application/x-mpegURL">
</video>
<script>
var player = videojs("sample");
player.textTracks()[0].mode = 'disabled';
</script>
"""
)

######################################################################
# Check the input streams
# -----------------------
#
# Firstly, let's list the available streams and its properties.
#

streamer = Streamer(src)
print(f"The number of source streams: {streamer.num_src_streams}\n")
for i in range(streamer.num_src_streams):
    print(f"STREAM {i}:")
    print("   ", streamer.get_src_stream_info(i))

######################################################################
# Find default streams
# --------------------
#
# Next, we configure output streams.
#
# To configure audio/video stream, you can use
# `add_basic_audio_stream` and `add_basic_video_stream` methods.
#
# There are also `add_audio_stream` and `add_video_stream` methods that
# gives granular control of the streams.
# We will cover them in the later section.
#
# All these method take the index of the source stream.
# You can manually specify it or find the best stream with
# `find_best_audio_stream` and `find_best_video_stream`.
# These methods use the heuristic implemented in ffmpeg to find
# the most suitable streams.

i_audio = streamer.find_best_audio_stream()
i_video = streamer.find_best_video_stream()

print("Best audio stream: ", i_audio)
print(" -", streamer.get_src_stream_info(i_audio))
print("Best video stream: ", i_video)
print(" -", streamer.get_src_stream_info(i_video))

######################################################################
# Configuring output streams
# --------------------------
#
# Now we configure the output stream.
#

streamer.add_basic_audio_stream(
    i_audio,
    frames_per_chunk=8000,
    sample_rate=8000,
)

streamer.add_basic_video_stream(
    i_video,
    frames_per_chunk=1,
    frame_rate=1,
    width=960,
    height=540,
    format="RGB",
)

######################################################################
# `frames_per_chunk` is the unit number of frames to be returned as a
# chunk.
#
# .. note::
#
#    When configuring multiple streams, to keep all the streams
#    synced, it is important to set the values of `frames_per_chunk`
#    and `sample_rate` in a way that the ratio are same across
#    streams.

######################################################################
# You can check the output streams with `get_out_stream_info` method.
#

for i in range(streamer.num_out_streams):
    print(streamer.get_out_stream_info(i))

######################################################################
# Iterate over the streams
# ------------------------
#
# For audio stream, the chunk Tensor will be the shape of
# `(frames_per_chunk, num_channels)`, and for video stream,
# it is `(frames_per_chunk, num_color_channels, height, width)`.
#

n_ite = 3
waveforms = []
images = []
for i, (waveform, image) in enumerate(streamer):
    print(f"Iteration: {i}, Audio: {waveform.shape}, Image: {image.shape}")
    waveforms.append(waveform)
    images.append(image)
    if i + 1 == n_ite:
        break

######################################################################
# Let's visualize what we received.
#

fig, axs = plt.subplots(nrows=n_ite, ncols=2)
for ax, waveform, image in zip(axs, waveforms, images):
    # plot waveform
    ax[0].plot(waveform)
    ax[0].set_ylim([-1, 1])
    ax[0].grid(True)
    plt.setp(ax[0].get_xticklabels(), visible=False)

    # plot image
    ax[1].imshow(image[0].permute(1, 2, 0))  # NCHW -> HWC
    ax[1].set_axis_off()
plt.show(block=False)

######################################################################
# Custom output stream
# --------------------
#
# When defining output stream, you can provide ``filter_desc``
# argument, which corresponds to ffmpeg's filter expression.
#
# https://ffmpeg.org/ffmpeg-filters.html
#
# .. note::
#
#    - When applying custom filters, the client code must convert
#      the audio/video stream to one of the formats that torchaudio
#      can convert to tensor format.
#      This can be achieved, for example, by applying
#      ``format=pix_fmts=rgb24`` to video stream and
#      ``aformat=sample_fmts=fltp`` to audio stream.
#    - Each output stream has separate filter graph. Therefore, it is
#      not possible to use different input/output streams for a
#      filter expression. However, it is possible to split one input
#      stream into multiple of them, and merge them later.
#

######################################################################
# Custom audio filters
# ~~~~~~~~~~~~~~~~~~~~
#

# fmt: off
descs = {
    "Original": "anull",
    "Highpass / lowpass filter": "highpass=f=200,lowpass=f=1000",
    "Robot": (
        "afftfilt="
        "real='hypot(re,im)*sin(0)':"
        "imag='hypot(re,im)*cos(0)':"
        "win_size=512:"
        "overlap=0.75"
    ),
    "Whisper": (
        "afftfilt="
        "real='hypot(re,im)*cos((random(0)*2-1)*2*3.14)':"
        "imag='hypot(re,im)*sin((random(1)*2-1)*2*3.14)':"
        "win_size=128:"
        "overlap=0.8"
    ),
}
# fmt: on

######################################################################
#

src = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
sample_rate = 8000

streamer = Streamer(src)
for desc in descs.values():
    streamer.add_audio_stream(
        0,
        frames_per_chunk=40000,
        filter_desc=f"aresample={sample_rate},{desc},aformat=sample_fmts=fltp",
    )

for chunks in streamer:
    break


def _display(i):
    fig, axs = plt.subplots(2, 1)
    waveform = chunks[i][:, 0]
    axs[0].plot(waveform)
    axs[0].set_title(list(descs)[i])
    axs[1].specgram(waveform, Fs=sample_rate)
    return IPython.display.Audio(chunks[i].T, rate=sample_rate)


######################################################################
# Original
# ^^^^^^^^
#

_display(0)

######################################################################
# Highpass / lowpass filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#

_display(1)

######################################################################
# FFT manipulation - robot
# ^^^^^^^^^^^^^^^^^^^^^^^^
#

_display(2)

######################################################################
# FFT manipulation - whisper
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#

_display(3)

######################################################################
# Custom video filters
# ~~~~~~~~~~~~~~~~~~~~
#

# fmt: off
descs = {
    # No effect
    "Original": "null",
    # Split the input stream and apply horizontal/vertical flip to the right half.
    "Mirror": (
        "split [main][tmp];"
        "[tmp] crop=iw/2:ih:0:0, hflip, vflip [flip];"
        "[main][flip] overlay=W/2:0"
    ),
    # Rotate image by randomly and fill the background with brown
    "Random Rotate": "rotate=angle=-random(1)*PI:fillcolor=brown",
    "Pixel manipulation": "geq=r='X/W*r(X,Y)':g='(1-X/W)*g(X,Y)':b='(H-Y)/H*b(X,Y)'"
}
# fmt: on

######################################################################
#

fps = 30
src = "https://devstreaming-cdn.apple.com/videos/streaming/examples/img_bipbop_adv_example_fmp4/master.m3u8"
streamer = Streamer(src)
for desc in descs.values():
    streamer.add_video_stream(
        i_video,
        frames_per_chunk=fps,
        filter_desc=f"fps={fps},{desc},format=pix_fmts=rgb24",
    )

for chunks in streamer:
    break

fig, axs = plt.subplots(len(descs), 3)
for images, ax, title in zip(chunks, axs, descs):
    for j in range(3):
        ax[j].imshow(images[10 * j + 1].permute(1, 2, 0))
        ax[j].set_axis_off()
        if j == 0:
            ax[j].set_title(title)
plt.show(block=False)
