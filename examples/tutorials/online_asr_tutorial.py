"""
Online ASR with Emformer RNN-T
==============================

**Author**: `Jeff Hwang <jeffhwang@fb.com>`__, `Moto Hira <moto@fb.com>`__

This tutorial shows how to use Emformer RNN-T and streaming API
to perform online speech recognition.

"""

######################################################################
# 1. Overview
# -----------
#
# Performing online speech recognition is composed of the following steps
#
# 1. Build the inference pipeline
#    Emformer RNN-T is composed of three components: feature extractor,
#    decoder and token processor.
# 2. Format the waveform into chunks of expected sizes.
# 3. Pass data through the pipeline.

######################################################################
# 2. Preparation
# --------------
#

######################################################################
#
# .. note::
#
#    The streaming API requires FFmpeg libraries (>=4.1).
#
#    If you are using Anaconda Python distribution,
#    ``conda install -c anaconda ffmpeg`` will install
#    the required libraries.
#
#    When running this tutorial in Google Colab, the following
#    command should do.
#
#    .. code::
#
#       !add-apt-repository -y ppa:savoury1/ffmpeg4
#       !apt-get -qq install -y ffmpeg

import IPython
import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

from torchaudio.prototype.io import Streamer

######################################################################
# 3. Construct the pipeline
# -------------------------
#
# Pre-trained model weights and related pipeline components are
# bundled as :py:func:`torchaudio.pipelines.RNNTBundle`.
#
# We use :py:func:`torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH`,
# which is a Emformer RNN-T model trained on LibriSpeech dataset.
#

bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH

feature_extractor = bundle.get_streaming_feature_extractor()
decoder = bundle.get_decoder()
token_processor = bundle.get_token_processor()

######################################################################
# Streaming inference works on input data with overlap.
# Emformer RNN-T expects right context like the following.
#
# .. image:: https://download.pytorch.org/torchaudio/tutorial-assets/emformer_rnnt_context.png
#
# The size of main segment and right context, along with
# the expected sample rate can be retrieved from bundle.
#

sample_rate = bundle.sample_rate
segment_length = bundle.segment_length * bundle.hop_length
context_length = bundle.right_context_length * bundle.hop_length

print(f"Sample rate: {sample_rate}")
print(f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)")
print(f"Right context: {context_length} frames ({context_length / sample_rate} seconds)")

######################################################################
# 4. Configure the audio stream
# -----------------------------
#
# Next, we configure the input audio stream using :py:func:`~torchaudio.prototype.io.Streamer`.
#
# For the detail of this API, please refer to the
# `Media Stream API tutorial <./streaming_api_tutorial.html>`__.
#

######################################################################
# The following audio file was originally published by LibriVox project,
# and it is in the public domain.
#
# https://librivox.org/great-pirate-stories-by-joseph-lewis-french/
#
# It was re-uploaded for the sake of the tutorial.
#
src = "https://download.pytorch.org/torchaudio/tutorial-assets/greatpiratestories_00_various.mp3"

streamer = Streamer(src)
streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=bundle.sample_rate)

print(streamer.get_src_stream_info(0))
print(streamer.get_out_stream_info(0))

######################################################################
# `Streamer` iterate the source media without overlap, so we make a
# helper structure that caches a chunk and return it with right context
# appended when the next chunk is given.
#


class ContextCacher:
    """Cache the previous chunk and combine it with the new chunk

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context


######################################################################
# 5. Run stream inference
# -----------------------
#
# Finally, we run the recognition.
#
# First, we initialize the stream iterator, context cacher, and
# state and hypothesis that are used by decoder to carry over the
# decoding state between inference calls.
#

cacher = ContextCacher(segment_length, context_length)

state, hypothesis = None, None

######################################################################
# Next we, run the inference.
#
# For the sake of better display, we create a helper function which
# processes the source stream up to the given times and call it
# repeatedly.
#


@torch.inference_mode()
def run_inference(num_iter=200):
    global state, hypothesis
    chunks = []
    for i, (chunk,) in enumerate(streamer.stream(), start=1):
        segment = cacher(chunk[:, 0])
        features, length = feature_extractor(segment)
        hypos, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
        hypothesis = hypos[0]
        transcript = token_processor(hypothesis.tokens, lstrip=False)
        print(transcript, end="", flush=True)

        chunks.append(chunk)
        if i == num_iter:
            break

    return IPython.display.Audio(torch.cat(chunks).T.numpy(), rate=bundle.sample_rate)


######################################################################
#

run_inference()

######################################################################
#

run_inference()

######################################################################
#

run_inference()

######################################################################
#

run_inference()

######################################################################
#

run_inference()

######################################################################
#

run_inference()

######################################################################
#

run_inference()
