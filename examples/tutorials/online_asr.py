"""
Online Speech Recognition with Emformer RNN-T
=============================================

"""

######################################################################
#

import torch
import torchaudio

from torchaudio.prototype.rnnt_pipeline import EMFORMER_RNNT_BASE_LIBRISPEECH
from torchaudio.prototype.ffmpeg import Streamer

bundle = EMFORMER_RNNT_BASE_LIBRISPEECH


decoder = bundle.get_decoder()
token_processor = bundle.get_token_processor()
streaming_feature_extractor = bundle.get_streaming_feature_extractor()

hop_length = bundle.hop_length
num_samples_segment = bundle.segment_length * hop_length
num_samples_segment_right_context = (
    num_samples_segment + bundle.right_context_length * hop_length
)

src = 'https://npr-ice.streamguys1.com/live.mp3'
streamer = Streamer(src)
streamer.add_basic_audio_stream(0, frames_per_chunk=num_samples_segment_right_context, sample_rate=bundle.sample_rate)

state, hypothesis = None, None
with torch.no_grad():
    for segment, in streamer:
        segment = segment.T[0]
        segment = torch.nn.functional.pad(
            segment, (0, num_samples_segment_right_context - len(segment)))
        features, length = streaming_feature_extractor(segment)
        hypos, state = decoder.infer(features, length, 10, state=state, hypothesis=hypothesis)
        hypothesis = hypos[0]
        transcript = token_processor(hypothesis.tokens)
        if transcript:
            print(transcript, end=" ", flush=True)
