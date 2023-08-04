"""
CTC forced alignment API tutorial
=================================

**Author**: `Xiaohui Zhang <xiaohuizhang@meta.com>`__


This tutorial shows how to align transcripts to speech using
:py:func:`torchaudio.functional.forced_align`
which was developed along the work of
`Scaling Speech Technology to 1,000+ Languages <https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/>`__.

The forced alignment is a process to align transcript with speech.
We cover the basics of forced alignment in `Forced Alignment with
Wav2Vec2 <./forced_alignment_tutorial.html>`__ with simplified
step-by-step Python implementations.

:py:func:`~torchaudio.functional.forced_align` has custom CPU and CUDA
implementations which are more performant than the vanilla Python
implementation above, and are more accurate.
It can also handle missing transcript with special <star> token.

For examples of aligning multiple languages, please refer to
`Forced alignment for multilingual data <./forced_alignment_for_multilingual_data_tutorial.html>`__.
"""

import torch
import torchaudio


print(torch.__version__)
print(torchaudio.__version__)

######################################################################
#

from dataclasses import dataclass
from typing import List

import IPython
import matplotlib.pyplot as plt

######################################################################
#

from torchaudio.functional import forced_align

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

######################################################################
# Preparation
# -----------
#
# First we prepare the speech data and the transcript we area going
# to use.
#

SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
TRANSCRIPT = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"

######################################################################
# Generating emissions and tokens
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# :py:func:`~torchaudio.functional.forced_align` takes emission and
# token sequences and outputs timestaps of the tokens and their scores.
#
# Emission reperesents the frame-wise probability distribution over
# tokens, and it can be obtained by passing waveform to an acoustic
# model.
# Tokens are numerical expression of transcripts. It can be obtained by
# simply mapping each character to the index of token list.
# The emission and the token sequences must be using the same set of tokens.
#
# We can use pre-trained Wav2Vec2 model to obtain emission from speech,
# and map transcript to tokens.
# Here, we use :py:data:`~torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`,
# which bandles pre-trained model weights with associated labels.
#

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emission, _ = model(waveform.to(device))
    emission = torch.log_softmax(emission, dim=-1)

num_frames = emission.size(1)


######################################################################
#


def plot_emission(emission):
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()


plot_emission(emission[0])

######################################################################
# We create a dictionary, which maps each label into token.

labels = bundle.get_labels()
DICTIONARY = {c: i for i, c in enumerate(labels)}

for k, v in DICTIONARY.items():
    print(f"{k}: {v}")

######################################################################
# converting transcript to tokens is as simple as

tokenized_transcript = [DICTIONARY[c] for c in TRANSCRIPT]

print(" ".join(str(t) for t in tokenized_transcript))

######################################################################
# Computing frame-level alignments
# --------------------------------
#
# Now we call TorchAudio’s forced alignment API to compute the
# frame-level alignment. For the detail of function signature, please
# refer to :py:func:`~torchaudio.functional.forced_align`.
#
#


def align(emission, tokens):
    alignments, scores = forced_align(
        emission,
        targets=torch.tensor([tokens], dtype=torch.int32, device=emission.device),
        input_lengths=torch.tensor([emission.size(1)], device=emission.device),
        target_lengths=torch.tensor([len(tokens)], device=emission.device),
        blank=0,
    )

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

######################################################################
# Now let's look at the output.
# Notice that the alignment is expressed in the frame cordinate of
# emission, which is different from the original waveform.

for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
    print(f"{i:3d}:\t{ali:2d} [{labels[ali]}], {score:.2f}")

######################################################################
#
# The ``Frame`` instance represents the most likely token at each frame
# with its confidence.
#
# When interpreting it, one must remember that the meaning of blank token
# and repeated token are context dependent.
#
# .. note::
#
#    When same token occured after blank tokens, it is not treated as
#    a repeat, but as a new occurrence.
#
#    .. code-block::
#
#       a a a b -> a b
#       a - - b -> a b
#       a a - b -> a b
#       a - a b -> a a b
#         ^^^       ^^^
#
# .. code-block::
#
#     29:  0 [-], 1.00
#     30:  7 [I], 1.00 # "I" starts and ends
#     31:  0 [-], 0.98 #
#     32:  0 [-], 1.00 #
#     33:  1 [|], 0.85 # "|" (word boundary) starts
#     34:  1 [|], 1.00 # "|" ends
#     35:  0 [-], 0.61 #
#     36:  8 [H], 1.00 # "H" starts and ends
#     37:  0 [-], 1.00 #
#     38:  4 [A], 1.00 # "A" starts and ends
#     39:  0 [-], 0.99 #
#     40: 11 [D], 0.92 # "D" starts and ends
#     41:  0 [-], 0.93 #
#     42:  1 [|], 0.98 # "|" starts
#     43:  1 [|], 1.00 # "|" ends
#     44:  3 [T], 1.00 # "T" starts
#     45:  3 [T], 0.90 # "T" ends
#     46:  8 [H], 1.00 # "H" starts and ends
#     47:  0 [-], 1.00 #

######################################################################
# Obtain token-level alignment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next step is to resolve the repetation. So that what alignment represents
# do not depend on previous alignments.
# From the outputs ``alignment``, we compute the following ``Span`` object,
# which explains what token (in transcript) is present at what time span.


@dataclass
class TokenSpan:
    index: int  # index of token in transcript
    start: int  # start time (inclusive)
    end: int  # end time (exclusive)
    score: float

    def __len__(self) -> int:
        return self.end - self.start


######################################################################
#


def merge_tokens(tokens, scores, blank=0) -> List[TokenSpan]:
    prev_token = blank
    i = start = -1
    spans = []
    for t, token in enumerate(tokens):
        if token != prev_token:
            if prev_token != blank:
                spans.append(TokenSpan(i, start, t, scores[start:t].mean().item()))
            if token != blank:
                i += 1
                start = t
            prev_token = token
    if prev_token != blank:
        spans.append(TokenSpan(i, start, len(tokens), scores[start:].mean().item()))
    return spans


######################################################################
#

token_spans = merge_tokens(aligned_tokens, alignment_scores)

print("Token\tTime\tScore")
for s in token_spans:
    print(f"{TRANSCRIPT[s.index]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")

######################################################################
# Visualization
# ~~~~~~~~~~~~~
#


def plot_scores(spans, scores, transcript):
    fig, ax = plt.subplots()
    ax.set_title("frame-level and token-level confidence scores")
    span_xs, span_hs, span_ws = [], [], []
    frame_xs, frame_hs = [], []
    for span in spans:
        token = transcript[span.index]
        if token != "|":
            span_xs.append((span.end + span.start) / 2 + 0.4)
            span_hs.append(span.score)
            span_ws.append(span.end - span.start)
            ax.annotate(token, (span.start + 0.8, -0.07), weight="bold")
            for t in range(span.start, span.end):
                frame_xs.append(t + 1)
                frame_hs.append(scores[t].item())
    ax.bar(span_xs, span_hs, width=span_ws, color="gray", alpha=0.5, edgecolor="black")
    ax.bar(frame_xs, frame_hs, width=0.5, alpha=0.5)

    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, axis="y")
    fig.tight_layout()


plot_scores(token_spans, alignment_scores, TRANSCRIPT)


######################################################################
# Obtain word-level alignments and confidence scores
# --------------------------------------------------
#

######################################################################
# Now let’s merge the token-level alignments and confidence scores to get
# word-level alignments and confidence scores. Then, finally, we verify
# the quality of word alignments by 1) plotting the word-level alignments
# and the waveform, 2) segmenting the original audio according to the
# alignments and listening to them.


@dataclass
class WordSpan:
    token_spans: List[TokenSpan]
    score: float


# Obtain word alignments from token alignments
def merge_words(token_spans, transcript, separator="|") -> List[WordSpan]:
    def _score(t_spans):
        return sum(s.score * len(s) for s in t_spans) / sum(len(s) for s in t_spans)

    words = []
    i = 0

    for j, span in enumerate(token_spans):
        if transcript[span.index] == separator:
            words.append(WordSpan(token_spans[i:j], _score(token_spans[i:j])))
            i = j + 1
    if i < len(token_spans):
        words.append(WordSpan(token_spans[i:], _score(token_spans[i:])))
    return words


word_spans = merge_words(token_spans, TRANSCRIPT)


######################################################################
# Visualization
# ~~~~~~~~~~~~~
#


def plot_alignments(waveform, word_spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    fig, ax = plt.subplots()

    ax.specgram(waveform[0], Fs=sample_rate)
    ratio = waveform.size(1) / sample_rate / num_frames
    for w_span in word_spans:
        t_spans = w_span.token_spans
        t0, t1 = t_spans[0].start, t_spans[-1].end
        ax.axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        ax.annotate(f"{w_span.score:.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span in t_spans:
            token = transcript[span.index]
            ax.annotate(token, (span.start * ratio, sample_rate * 0.53), annotation_clip=False)

    ax.set_xlabel("time [second]")
    ax.set_xlim([0, None])
    fig.tight_layout()


plot_alignments(waveform, word_spans, num_frames, TRANSCRIPT)


######################################################################


def preview_word(waveform, word_span, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    t0 = word_span.token_spans[0].start
    t1 = word_span.token_spans[-1].end
    x0 = int(ratio * t0)
    x1 = int(ratio * t1)
    tokens = "".join(transcript[t.index] for t in word_span.token_spans)
    print(f"{tokens} ({word_span.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


######################################################################

# Generate the audio for each segment
print(TRANSCRIPT)
IPython.display.Audio(SPEECH_FILE)

######################################################################
#

preview_word(waveform, word_spans[0], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[1], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[2], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[3], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[4], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[5], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[6], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[7], num_frames, TRANSCRIPT)

######################################################################
#

preview_word(waveform, word_spans[8], num_frames, TRANSCRIPT)


######################################################################
# Advanced: Handling transcripts with ``<star>`` token
# ----------------------------------------------------
#
# Now let’s look at when the transcript is partially missing, how can we
# improve alignment quality using the ``<star>`` token, which is capable of modeling
# any token.
#
# Here we use the same English example as used above. But we remove the
# beginning text ``“i had that curiosity beside me at”`` from the transcript.
# Aligning audio with such transcript results in wrong alignments of the
# existing word “this”. However, this issue can be mitigated by using the
# ``<star>`` token to model the missing text.
#

######################################################################
# First, we extend the dictionary to include the ``<star>`` token.

DICTIONARY["*"] = len(DICTIONARY)

######################################################################
# Next, we extend the emission tensor with the extra dimension
# corresponding to the ``<star>`` token.
#

star_dim = torch.zeros((1, num_frames, 1), device=device)
emission = torch.cat((emission, star_dim), 2)

assert len(DICTIONARY) == emission.shape[2]

plot_emission(emission[0])

######################################################################
# The following function combines all the processes, and compute
# word segments from emission in one-go.


def compute_alignments(emission, transcript, dictionary):
    tokens = [dictionary[c] for c in transcript]
    alignment, scores = align(emission, tokens)
    token_spans = merge_tokens(alignment, scores)
    word_spans = merge_words(token_spans, transcript)
    return word_spans


######################################################################
# **Original**

word_spans = compute_alignments(emission, TRANSCRIPT, DICTIONARY)
plot_alignments(waveform, word_spans, num_frames, TRANSCRIPT)

######################################################################
# **With <star> token**
#
# Now we replace the first part of the transcript with the ``<star>`` token.

transcript = "*|THIS|MOMENT"
word_spans = compute_alignments(emission, transcript, DICTIONARY)
plot_alignments(waveform, word_spans, num_frames, transcript)

######################################################################
#

preview_word(waveform, word_spans[1], num_frames, transcript)

######################################################################
#

preview_word(waveform, word_spans[2], num_frames, transcript)

######################################################################
#

######################################################################
# **Without <star> token**
#
# As a comparison, the following aligns the partial transcript
# without using ``<star>`` token.
# It demonstrates the effect of ``<star>`` token for dealing with deletion errors.

transcript = "THIS|MOMENT"
word_spans = compute_alignments(emission, transcript, DICTIONARY)
plot_alignments(waveform, word_spans, num_frames, transcript)

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we looked at how to use torchaudio’s forced alignment
# API to align and segment speech files, and demonstrated one advanced usage:
# How introducing a ``<star>`` token could improve alignment accuracy when
# transcription errors exist.
#


######################################################################
# Acknowledgement
# ---------------
#
# Thanks to `Vineel Pratap <vineelkpratap@meta.com>`__ and `Zhaoheng
# Ni <zni@meta.com>`__ for working on the forced aligner API, and `Moto
# Hira <moto@meta.com>`__ for providing alignment merging and
# visualization utilities.
#
