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


######################################################################
#


def plot_emission(emission):
    plt.imshow(emission.cpu().T)
    plt.title("Frame-wise class probabilities")
    plt.xlabel("Time")
    plt.ylabel("Labels")
    plt.tight_layout()


plot_emission(emission[0])

######################################################################
# We create a dictionary, which maps each label into token.

labels = bundle.get_labels()
DICTIONARY = {c: i for i, c in enumerate(labels)}

for k, v in DICTIONARY.items():
    print(f"{k}: {v}")

######################################################################
# converting transcript to tokens is as simple as

tokens = [DICTIONARY[c] for c in TRANSCRIPT]

print(" ".join(str(t) for t in tokens))

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

    scores = scores.exp()  # convert back to probability
    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    return alignments.tolist(), scores.tolist()


frame_alignment, frame_scores = align(emission, tokens)

######################################################################
# Now let's look at the output.
# Notice that the alignment is expressed in the frame cordinate of
# emission, which is different from the original waveform.

for i, (ali, score) in enumerate(zip(frame_alignment, frame_scores)):
    print(f"{i:3d}: {ali:2d} [{labels[ali]}], {score:.2f}")

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
#     30:  7 [I], 1.00 # Start of "I"
#     31:  0 [-], 0.98 #               repeat (blank token)
#     32:  0 [-], 1.00 #               repeat (blank token)
#     33:  1 [|], 0.85 # Start of "|" (word boundary)
#     34:  1 [|], 1.00 #               repeat (same token)
#     35:  0 [-], 0.61 #               repeat (blank token)
#     36:  8 [H], 1.00 # Start of "H"
#     37:  0 [-], 1.00 #               repeat (blank token)
#     38:  4 [A], 1.00 # Start of "A"
#     39:  0 [-], 0.99 #               repeat (blank token)
#     40: 11 [D], 0.92 # Start of "D"
#     41:  0 [-], 0.93 #               repeat (blank token)
#     42:  1 [|], 0.98 # Start of "|"
#     43:  1 [|], 1.00 #               repeat (same token)
#     44:  3 [T], 1.00 # Start of "T"
#     45:  3 [T], 0.90 #               repeat (same token)
#     46:  8 [H], 1.00 # Start of "H"
#     47:  0 [-], 1.00 #               repeat (blank token)

######################################################################
# Resolve blank and repeated tokens
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next step is to resolve the repetation. So that what alignment represents
# do not depend on previous alignments.
# From the outputs ``alignment`` and ``scores``, we generate a
# list called ``frames`` storing information of all frames aligned to
# non-blank tokens.
#
# Each element contains the following
#
# - ``token_index``: the aligned token’s index **in the transcript**
# - ``time_index``: the current frame’s index in emission
# - ``score``: scores of the current frame.
#
# ``token_index`` is the index of each token in the transcript,
# i.e. the current frame aligns to the N-th character from the transcript.


@dataclass
class Frame:
    token_index: int
    time_index: int
    score: float


######################################################################
#


def obtain_token_level_alignments(alignments, scores) -> List[Frame]:
    assert len(alignments) == len(scores)

    token_index = -1
    prev_hyp = 0
    frames = []
    for i, (ali, score) in enumerate(zip(alignments, scores)):
        if ali == 0:
            prev_hyp = 0
            continue

        if ali != prev_hyp:
            token_index += 1
        frames.append(Frame(token_index, i, score))
        prev_hyp = ali
    return frames


######################################################################
#

frames = obtain_token_level_alignments(frame_alignment, frame_scores)

print("Time\tLabel\tScore")
for f in frames:
    print(f"{f.time_index:3d}\t{TRANSCRIPT[f.token_index]}\t{f.score:.2f}")


######################################################################
# Obtain token-level alignments and confidence scores
# ---------------------------------------------------
#
# The frame-level alignments contains repetations for the same labels.
# Another format “token-level alignment”, which specifies the aligned
# frame ranges for each transcript token, contains the same information,
# while being more convenient to apply to some downstream tasks
# (e.g. computing word-level alignments).
#
# Now we demonstrate how to obtain token-level alignments and confidence
# scores by simply merging frame-level alignments and averaging
# frame-level confidence scores.
#

######################################################################
# The following class represents the label, its score and the time span
# of its occurance.
#


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label:2s} ({self.score:4.2f}): [{self.start:4d}, {self.end:4d})"

    def __len__(self):
        return self.end - self.start


######################################################################
#


def merge_repeats(frames, transcript):
    transcript_nospace = transcript.replace(" ", "")
    i1, i2 = 0, 0
    segments = []
    while i1 < len(frames):
        while i2 < len(frames) and frames[i1].token_index == frames[i2].token_index:
            i2 += 1
        score = sum(frames[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript_nospace[frames[i1].token_index],
                frames[i1].time_index,
                frames[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


######################################################################
#
segments = merge_repeats(frames, TRANSCRIPT)
for seg in segments:
    print(seg)


######################################################################
# Visualization
# ~~~~~~~~~~~~~
#


def plot_label_prob(segments, transcript):
    fig, ax = plt.subplots()

    ax.set_title("frame-level and token-level confidence scores")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
    ax.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in frames:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax.bar(xs, hs, width=0.5, alpha=0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, axis="y")
    fig.tight_layout()


plot_label_prob(segments, TRANSCRIPT)


######################################################################
# From the visualized scores, we can see that, for tokens spanning over
# more multiple frames, e.g. “T” in “THAT, the token-level confidence
# score is the average of frame-level confidence scores. To make this
# clearer, we don’t plot confidence scores for blank frames, which was
# plotted in the”Label probability with and without repeatation” figure in
# the previous tutorial
# `Forced Alignment with Wav2Vec2 <./forced_alignment_tutorial.html>`__.
#

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


# Obtain word alignments from token alignments
def merge_words(transcript, segments, separator=" "):
    words = []
    i1, i2, i3 = 0, 0, 0
    while i3 < len(transcript):
        if i3 == len(transcript) - 1 or transcript[i3] == separator:
            if i1 != i2:
                if i3 == len(transcript) - 1:
                    i2 += 1
                if separator == "|":
                    # s is the number of separators (counted as a valid modeling unit) we've seen
                    s = len(words)
                else:
                    s = 0
                segs = segments[i1 + s : i2 + s]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * len(seg) for seg in segs) / sum(len(seg) for seg in segs)
                words.append(Segment(word, segments[i1 + s].start, segments[i2 + s - 1].end, score))
            i1 = i2
        else:
            i2 += 1
        i3 += 1
    return words


word_segments = merge_words(TRANSCRIPT, segments, "|")


######################################################################
# Visualization
# ~~~~~~~~~~~~~
#


def plot_alignments(waveform, emission, segments, word_segments, sample_rate=bundle.sample_rate):
    fig, ax = plt.subplots()

    ax.specgram(waveform[0], Fs=sample_rate)

    # The original waveform
    ratio = waveform.size(1) / sample_rate / emission.size(1)
    for word in word_segments:
        t0, t1 = ratio * word.start, ratio * word.end
        ax.axvspan(t0, t1, facecolor="None", hatch="/", edgecolor="white")
        ax.annotate(f"{word.score:.2f}", (t0, sample_rate * 0.51), annotation_clip=False)

    for seg in segments:
        if seg.label != "|":
            ax.annotate(seg.label, (seg.start * ratio, sample_rate * 0.53), annotation_clip=False)

    ax.set_xlabel("time [second]")
    fig.tight_layout()


plot_alignments(waveform, emission, segments, word_segments)


######################################################################


def display_segment(i, waveform, word_segments, frame_alignment, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / len(frame_alignment)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


######################################################################

# Generate the audio for each segment
print(TRANSCRIPT)
IPython.display.Audio(SPEECH_FILE)

######################################################################
#

display_segment(0, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(1, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(2, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(3, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(4, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(5, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(6, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(7, waveform, word_segments, frame_alignment)

######################################################################
#

display_segment(8, waveform, word_segments, frame_alignment)


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

extra_dim = torch.zeros(emission.shape[0], emission.shape[1], 1, device=device)
emission = torch.cat((emission, extra_dim), 2)

assert len(DICTIONARY) == emission.shape[2]


######################################################################
# The following function combines all the processes, and compute
# word segments from emission in one-go.


def compute_and_plot_alignments(transcript, dictionary, emission, waveform):
    tokens = [dictionary[c] for c in transcript]
    alignment, scores = align(emission, tokens)
    frames = obtain_token_level_alignments(alignment, scores)
    segments = merge_repeats(frames, transcript)
    word_segments = merge_words(transcript, segments, "|")
    plot_alignments(waveform, emission, segments, word_segments)
    plt.xlim([0, None])


######################################################################
# **Original**

compute_and_plot_alignments(TRANSCRIPT, DICTIONARY, emission, waveform)

######################################################################
# **With <star> token**
#
# Now we replace the first part of the transcript with the ``<star>`` token.

compute_and_plot_alignments("*|THIS|MOMENT", DICTIONARY, emission, waveform)

######################################################################
# **Without <star> token**
#
# As a comparison, the following aligns the partial transcript
# without using ``<star>`` token.
# It demonstrates the effect of ``<star>`` token for dealing with deletion errors.

compute_and_plot_alignments("THIS|MOMENT", DICTIONARY, emission, waveform)

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
