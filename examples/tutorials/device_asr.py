"""
Device ASR with Emformer RNN-T
==============================

**Author**: `Moto Hira <moto@meta.com>`__, `Jeff Hwang <jeffhwang@meta.com>`__.

This tutorial shows how to use Emformer RNN-T and streaming API
to perform speech recognition on a streaming device input, i.e. microphone
on laptop.
"""

######################################################################
#
# .. note::
#
#    This tutorial requires FFmpeg libraries.
#    Please refer to :ref:`FFmpeg dependency <ffmpeg_dependency>` for
#    the detail.
#
# .. note::
#
#    This tutorial was tested on MacBook Pro and Dynabook with Windows 10.
#
#    This tutorial does NOT work on Google Colab because the server running
#    this tutorial does not have a microphone that you can talk to.

######################################################################
# 1. Overview
# -----------
#
# We use streaming API to fetch audio from audio device (microphone)
# chunk by chunk, then run inference using Emformer RNN-T.
#
# For the basic usage of the streaming API and Emformer RNN-T
# please refer to
# `StreamReader Basic Usage <./streamreader_basic_tutorial.html>`__ and
# `Online ASR with Emformer RNN-T <./online_asr_tutorial.html>`__.
#

######################################################################
# 2. Checking the supported devices
# ---------------------------------
#
# Firstly, we need to check the devices that Streaming API can access,
# and figure out the arguments (``src`` and ``format``) we need to pass
# to :py:func:`~torchaudio.io.StreamReader` class.
#
# We use ``ffmpeg`` command for this. ``ffmpeg`` abstracts away the
# difference of underlying hardware implementations, but the expected
# value for ``format`` varies across OS and each ``format`` defines
# different syntax for ``src``.
#
# The details of supported ``format`` values and ``src`` syntax can
# be found in https://ffmpeg.org/ffmpeg-devices.html.
#
# For macOS, the following command will list the available devices.
#
# .. code::
#
#    $ ffmpeg -f avfoundation -list_devices true -i dummy
#    ...
#    [AVFoundation indev @ 0x126e049d0] AVFoundation video devices:
#    [AVFoundation indev @ 0x126e049d0] [0] FaceTime HD Camera
#    [AVFoundation indev @ 0x126e049d0] [1] Capture screen 0
#    [AVFoundation indev @ 0x126e049d0] AVFoundation audio devices:
#    [AVFoundation indev @ 0x126e049d0] [0] ZoomAudioDevice
#    [AVFoundation indev @ 0x126e049d0] [1] MacBook Pro Microphone
#
# We will use the following values for Streaming API.
#
# .. code::
#
#    StreamReader(
#        src = ":1",  # no video, audio from device 1, "MacBook Pro Microphone"
#        format = "avfoundation",
#    )

######################################################################
#
# For Windows, ``dshow`` device should work.
#
# .. code::
#
#    > ffmpeg -f dshow -list_devices true -i dummy
#    ...
#    [dshow @ 000001adcabb02c0] DirectShow video devices (some may be both video and audio devices)
#    [dshow @ 000001adcabb02c0]  "TOSHIBA Web Camera - FHD"
#    [dshow @ 000001adcabb02c0]     Alternative name "@device_pnp_\\?\usb#vid_10f1&pid_1a42&mi_00#7&27d916e6&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\global"
#    [dshow @ 000001adcabb02c0] DirectShow audio devices
#    [dshow @ 000001adcabb02c0]  "... (Realtek High Definition Audio)"
#    [dshow @ 000001adcabb02c0]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{BF2B8AE1-10B8-4CA4-A0DC-D02E18A56177}"
#
# In the above case, the following value can be used to stream from microphone.
#
# .. code::
#
#    StreamReader(
#        src = "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{BF2B8AE1-10B8-4CA4-A0DC-D02E18A56177}",
#        format = "dshow",
#    )
#

######################################################################
# 3. Data acquisition
# -------------------
#
# Streaming audio from microphone input requires properly timing data
# acquisition. Failing to do so may introduce discontinuities in the
# data stream.
#
# For this reason, we will run the data acquisition in a subprocess.
#
# Firstly, we create a helper function that encapsulates the whole
# process executed in the subprocess.
#
# This function initializes the streaming API, acquires data then
# puts it in a queue, which the main process is watching.
#

import torch
import torchaudio


# The data acquisition process will stop after this number of steps.
# This eliminates the need of process synchronization and makes this
# tutorial simple.
NUM_ITER = 100


def stream(q, format, src, segment_length, sample_rate):
    from torchaudio.io import StreamReader

    print("Building StreamReader...")
    streamer = StreamReader(src, format=format)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

    print(streamer.get_src_stream_info(0))
    print(streamer.get_out_stream_info(0))

    print("Streaming...")
    print()
    stream_iterator = streamer.stream(timeout=-1, backoff=1.0)
    for _ in range(NUM_ITER):
        (chunk,) = next(stream_iterator)
        q.put(chunk)


######################################################################
#
# The notable difference from the non-device streaming is that,
# we provide ``timeout`` and ``backoff`` parameters to ``stream`` method.
#
# When acquiring data, if the rate of acquisition requests is higher
# than that at which the hardware can prepare the data, then
# the underlying implementation reports special error code, and expects
# client code to retry.
#
# Precise timing is the key for smooth streaming. Reporting this error
# from low-level implementation all the way back to Python layer,
# before retrying adds undesired overhead.
# For this reason, the retry behavior is implemented in C++ layer, and
# ``timeout`` and ``backoff`` parameters allow client code to control the
# behavior.
#
# For the detail of ``timeout`` and ``backoff`` parameters, please refer
# to the documentation of
# :py:meth:`~torchaudio.io.StreamReader.stream` method.
#
# .. note::
#
#    The proper value of ``backoff`` depends on the system configuration.
#    One way to see if ``backoff`` value is appropriate is to save the
#    series of acquired chunks as a continuous audio and listen to it.
#    If ``backoff`` value is too large, then the data stream is discontinuous.
#    The resulting audio sounds sped up.
#    If ``backoff`` value is too small or zero, the audio stream is fine,
#    but the data acquisition process enters busy-waiting state, and
#    this increases the CPU consumption.
#

######################################################################
# 4. Building inference pipeline
# ------------------------------
#
# The next step is to create components required for inference.
#
# This is the same process as
# `Online ASR with Emformer RNN-T <./online_asr_tutorial.html>`__.
#


class Pipeline:
    """Build inference pipeline from RNNTBundle.

    Args:
        bundle (torchaudio.pipelines.RNNTBundle): Bundle object
        beam_width (int): Beam size of beam search decoder.
    """

    def __init__(self, bundle: torchaudio.pipelines.RNNTBundle, beam_width: int = 10):
        self.bundle = bundle
        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()

        self.beam_width = beam_width

        self.state = None
        self.hypotheses = None

    def infer(self, segment: torch.Tensor) -> str:
        """Perform streaming inference"""
        features, length = self.feature_extractor(segment)
        self.hypotheses, self.state = self.decoder.infer(
            features, length, self.beam_width, state=self.state, hypothesis=self.hypotheses
        )
        transcript = self.token_processor(self.hypotheses[0][0], lstrip=False)
        return transcript


######################################################################
#


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

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
# 5. The main process
# -------------------
#
# The execution flow of the main process is as follows:
#
# 1. Initialize the inference pipeline.
# 2. Launch data acquisition subprocess.
# 3. Run inference.
# 4. Clean up
#
# .. note::
#
#    As the data acquisition subprocess will be launched with `"spawn"`
#    method, all the code on global scope are executed on the subprocess
#    as well.
#
#    We want to instantiate pipeline only in the main process,
#    so we put them in a function and invoke it within
#    `__name__ == "__main__"` guard.
#


def main(device, src, bundle):
    print(torch.__version__)
    print(torchaudio.__version__)

    print("Building pipeline...")
    pipeline = Pipeline(bundle)

    sample_rate = bundle.sample_rate
    segment_length = bundle.segment_length * bundle.hop_length
    context_length = bundle.right_context_length * bundle.hop_length

    print(f"Sample rate: {sample_rate}")
    print(f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)")
    print(f"Right context: {context_length} frames ({context_length / sample_rate} seconds)")

    cacher = ContextCacher(segment_length, context_length)

    @torch.inference_mode()
    def infer():
        for _ in range(NUM_ITER):
            chunk = q.get()
            segment = cacher(chunk[:, 0])
            transcript = pipeline.infer(segment)
            print(transcript, end="\r", flush=True)

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=stream, args=(q, device, src, segment_length, sample_rate))
    p.start()
    infer()
    p.join()


if __name__ == "__main__":
    main(
        device="avfoundation",
        src=":1",
        bundle=torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH,
    )

######################################################################
#
# .. code::
#
#    Building pipeline...
#    Sample rate: 16000
#    Main segment: 2560 frames (0.16 seconds)
#    Right context: 640 frames (0.04 seconds)
#    Building StreamReader...
#    SourceAudioStream(media_type='audio', codec='pcm_f32le', codec_long_name='PCM 32-bit floating point little-endian', format='flt', bit_rate=1536000, sample_rate=48000.0, num_channels=1)
#    OutputStream(source_index=0, filter_description='aresample=16000,aformat=sample_fmts=fltp')
#    Streaming...
#
#    hello world
#

######################################################################
#
# Tag: :obj:`torchaudio.io`
#
