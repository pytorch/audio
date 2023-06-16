"""
CTC forced alignment API tutorial
=================================

**Author**: `Xiaohui Zhang <xiaohuizhang@meta.com>`__


This tutorial shows how to align transcripts to speech with
``torchaudio``'s CTC forced alignment API proposed in the paper
`“Scaling Speech Technology to 1,000+
Languages” <https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/>`__,
and one advanced usage, i.e. dealing with transcription errors with a <star> token.

Though there’s some overlap in visualization
diagrams, the scope here is different from the `“Forced Alignment with
Wav2Vec2” <https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html>`__
tutorial, which focuses on a step-by-step demonstration of the forced
alignment generation algorithm (without using an API) described in the
`paper <https://arxiv.org/abs/2007.09127>`__ with a Wav2Vec2 model.

"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


try:
    from torchaudio.functional import forced_align
except ModuleNotFoundError:
    print(
        "Failed to import the forced alignment API. "
        "Please install torchaudio nightly builds. "
        "Please refer to https://pytorch.org/get-started/locally "
        "for instructions to install a nightly build."
    )
    raise

######################################################################
# I. Basic usages
# ---------------
#
# In this section, we cover the following content:
#
# 1. Generate frame-wise class probabilites from audio waveform from a CTC
#    acoustic model.
# 2. Compute frame-level alignments using TorchAudio’s forced alignment
#    API.
# 3. Obtain token-level alignments from frame-level alignments.
# 4. Obtain word-level alignments from token-level alignments.
#


######################################################################
# Preparation
# ~~~~~~~~~~~
#
# First we import the necessary packages, and fetch data that we work on.
#

# %matplotlib inline
from dataclasses import dataclass

import IPython
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)

SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
sample_rate = 16000


######################################################################
# Generate frame-wise class posteriors from a CTC acoustic model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The first step is to generate the class probabilities (i.e. posteriors)
# of each audio frame using a CTC model.
# Here we use :py:func:`torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`.
#

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
labels = bundle.get_labels()
with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

emission = emissions[0].cpu().detach()
dictionary = {c: i for i, c in enumerate(labels)}

print(dictionary)


######################################################################
# Visualization
# ^^^^^^^^^^^^^
#

plt.imshow(emission.T)
plt.colorbar()
plt.title("Frame-wise class probabilities")
plt.xlabel("Time")
plt.ylabel("Labels")
plt.show()


######################################################################
# Computing frame-level alignments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Then we call TorchAudio’s forced alignment API to compute the
# frame-level alignment between each audio frame and each token in the
# transcript. We first explain the inputs and outputs of the API
# ``functional.forced_align``. Note that this API works on both CPU and
# GPU. In the current tutorial we demonstrate it on CPU.
#
# **Inputs**:
#
# ``emission``: a 2D tensor of size :math:`T \times N`, where :math:`T` is
# the number of frames (after sub-sampling by the acoustic model, if any),
# and :math:`N` is the vocabulary size.
#
# ``targets``: a 1D tensor vector of size :math:`M`, where :math:`M` is
# the length of the transcript, and each element is a token ID looked up
# from the vocabulary. For example, the ``targets`` tensor repsenting the
# transcript “i had…” is :math:`[5, 18, 4, 16, ...]`.
#
# ``input lengths``: :math:`T`.
#
# ``target lengths``: :math:`M`.
#
# **Outputs**:
#
# ``frame_alignment``: a 1D tensor of size :math:`T` storing the aligned
# token index (looked up from the vocabulary) of each frame, e.g. for the
# segment corresponding to “i had” in the given example , the
# frame_alignment is
# :math:`[...0, 0, 5, 0, 0, 18, 18, 4, 0, 0, 0, 16,...]`, where :math:`0`
# represents the blank symbol.
#
# ``frame_scores``: a 1D tensor of size :math:`T` storing the confidence
# score (0 to 1) for each each frame. For each frame, the score should be
# close to one if the alignment quality is good.
#
# From the outputs ``frame_alignment`` and ``frame_scores``, we generate a
# list called ``frames`` storing information of all frames aligned to
# non-blank tokens. Each element contains 1) token_index: the aligned
# token’s index in the transcript 2) time_index: the current frame’s index
# in the input audio (or more precisely, the row dimension of the emission
# matrix) 3) the confidence scores of the current frame.
#
# For the given example, the first few elements of the list ``frames``
# corresponding to “i had” looks as the following:
#
# ``Frame(token_index=0, time_index=32, score=0.9994410872459412)``
#
# ``Frame(token_index=1, time_index=35, score=0.9980823993682861)``
#
# ``Frame(token_index=1, time_index=36, score=0.9295750260353088)``
#
# ``Frame(token_index=2, time_index=37, score=0.9997448325157166)``
#
# ``Frame(token_index=3, time_index=41, score=0.9991760849952698)``
#
# ``...``
#
# The interpretation is:
#
# The token with index :math:`0` in the transcript, i.e. “i”, is aligned
# to the :math:`32`\ th audio frame, with confidence :math:`0.9994`. The
# token with index :math:`1` in the transcript, i.e. “h”, is aligned to
# the :math:`35`\ th and :math:`36`\ th audio frames, with confidence
# :math:`0.9981` and :math:`0.9296` respectively. The token with index
# :math:`2` in the transcript, i.e. “a”, is aligned to the :math:`35`\ th
# and :math:`36`\ th audio frames, with confidence :math:`0.9997`. The
# token with index :math:`3` in the transcript, i.e. “d”, is aligned to
# the :math:`41`\ th audio frame, with confidence :math:`0.9992`.
#
# From such information stored in the ``frames`` list, we’ll compute
# token-level and word-level alignments easily.
#


@dataclass
class Frame:
    # This is the index of each token in the transcript,
    # i.e. the current frame aligns to the N-th character from the transcript.
    token_index: int
    time_index: int
    score: float


def compute_alignments(transcript, dictionary, emission):
    frames = []
    tokens = [dictionary[c] for c in transcript.replace(" ", "")]

    targets = torch.tensor(tokens, dtype=torch.int32)
    input_lengths = torch.tensor(emission.shape[0])
    target_lengths = torch.tensor(targets.shape[0])

    # This is the key step, where we call the forced alignment API functional.forced_align to compute alignments.
    frame_alignment, frame_scores = forced_align(emission, targets, input_lengths, target_lengths, 0)

    assert len(frame_alignment) == input_lengths.item()
    assert len(targets) == target_lengths.item()

    token_index = -1
    prev_hyp = 0
    for i in range(len(frame_alignment)):
        if frame_alignment[i].item() == 0:
            prev_hyp = 0
            continue

        if frame_alignment[i].item() != prev_hyp:
            token_index += 1
        frames.append(Frame(token_index, i, frame_scores[i].exp().item()))
        prev_hyp = frame_alignment[i].item()
    return frames, frame_alignment, frame_scores


transcript = "I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT"
frames, frame_alignment, frame_scores = compute_alignments(transcript, dictionary, emission)


######################################################################
# Obtain token-level alignments and confidence scores
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# The frame-level alignments contains repetations for the same labels.
# Another format “token-level alignment”, which specifies the aligned
# frame ranges for each transcript token, contains the same information,
# while being more convenient to apply to some downstream tasks
# (e.g. computing word-level alignments).
#
# Now we demonstrate how to obtain token-level alignments and confidence
# scores by simply merging frame-level alignments and averaging
# frame-level confidence scores.
#


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


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


segments = merge_repeats(frames, transcript)
for seg in segments:
    print(seg)


######################################################################
# Visualization
# ^^^^^^^^^^^^^
#


def plot_label_prob(segments, transcript):
    fig, ax2 = plt.subplots(figsize=(16, 4))

    ax2.set_title("frame-level and token-level confidence scores")
    xs, hs, ws = [], [], []
    for seg in segments:
        if seg.label != "|":
            xs.append((seg.end + seg.start) / 2 + 0.4)
            hs.append(seg.score)
            ws.append(seg.end - seg.start)
            ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
    ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

    xs, hs = [], []
    for p in frames:
        label = transcript[p.token_index]
        if label != "|":
            xs.append(p.time_index + 1)
            hs.append(p.score)

    ax2.bar(xs, hs, width=0.5, alpha=0.5)
    ax2.axhline(0, color="black")
    ax2.set_ylim(-0.1, 1.1)


plot_label_prob(segments, transcript)
plt.tight_layout()
plt.show()


######################################################################
# From the visualized scores, we can see that, for tokens spanning over
# more multiple frames, e.g. “T” in “THAT, the token-level confidence
# score is the average of frame-level confidence scores. To make this
# clearer, we don’t plot confidence scores for blank frames, which was
# plotted in the”Label probability with and without repeatation” figure in
# the previous tutorial `“Forced Alignment with
# Wav2Vec2” <https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html>`__.
#
# Obtain word-level alignments and confidence scores
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1 + s].start, segments[i2 + s - 1].end, score))
            i1 = i2
        else:
            i2 += 1
        i3 += 1
    return words


word_segments = merge_words(transcript, segments, "|")


######################################################################
# Visualization
# ^^^^^^^^^^^^^
#


def plot_alignments(segments, word_segments, waveform, input_lengths, scale=10):
    fig, ax2 = plt.subplots(figsize=(64, 12))
    plt.rcParams.update({"font.size": 30})

    # The original waveform
    ratio = waveform.size(0) / input_lengths
    ax2.plot(waveform)
    ax2.set_ylim(-1.0 * scale, 1.0 * scale)
    ax2.set_xlim(0, waveform.size(-1))

    for word in word_segments:
        x0 = ratio * word.start
        x1 = ratio * word.end
        ax2.axvspan(x0, x1, alpha=0.1, color="red")
        ax2.annotate(f"{word.score:.2f}", (x0, 0.8 * scale))

    for seg in segments:
        if seg.label != "|":
            ax2.annotate(seg.label, (seg.start * ratio, 0.9 * scale))

    xticks = ax2.get_xticks()
    plt.xticks(xticks, xticks / sample_rate, fontsize=50)
    ax2.set_xlabel("time [second]", fontsize=40)
    ax2.set_yticks([])


plot_alignments(
    segments,
    word_segments,
    waveform[0],
    emission.shape[0],
    1,
)
plt.show()


######################################################################


# A trick to embed the resulting audio to the generated file.
# `IPython.display.Audio` has to be the last call in a cell,
# and there should be only one call par cell.
def display_segment(i, waveform, word_segments, frame_alignment):
    ratio = waveform.size(1) / len(frame_alignment)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


# Generate the audio for each segment
print(transcript)
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
# II. Advanced usage: Dealing with missing transcripts using the <star> token
# ---------------------------------------------------------------------------
#
# Now let’s look at when the transcript is partially missing, how can we
# improve alignment quality using the <star> token, which is capable of modeling
# any token.
#
# Here we use the same English example as used above. But we remove the
# beginning text “i had that curiosity beside me at” from the transcript.
# Aligning audio with such transcript results in wrong alignments of the
# existing word “this”. However, this issue can be mitigated by using the
# <star> token to model the missing text.
#

# Reload the emission tensor in order to add the extra dimension corresponding to the <star> token.
with torch.inference_mode():
    waveform, _ = torchaudio.load(SPEECH_FILE)
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1)

    # Append the extra dimension corresponding to the <star> token
    extra_dim = torch.zeros(emissions.shape[0], emissions.shape[1], 1)
    emissions = torch.cat((emissions.cpu(), extra_dim), 2)
    emission = emissions[0].detach()

# Extend the dictionary to include the <star> token.
dictionary["*"] = 29

assert len(dictionary) == emission.shape[1]


def compute_and_plot_alignments(transcript, dictionary, emission, waveform):
    frames, frame_alignment, _ = compute_alignments(transcript, dictionary, emission)
    segments = merge_repeats(frames, transcript)
    word_segments = merge_words(transcript, segments, "|")
    plot_alignments(segments, word_segments, waveform[0], emission.shape[0], 1)
    plt.show()
    return word_segments, frame_alignment


# original:
word_segments, frame_alignment = compute_and_plot_alignments(transcript, dictionary, emission, waveform)

######################################################################

# Demonstrate the effect of <star> token for dealing with deletion errors
# ("i had that curiosity beside me at" missing from the transcript):
transcript = "THIS|MOMENT"
word_segments, frame_alignment = compute_and_plot_alignments(transcript, dictionary, emission, waveform)

######################################################################

# Replacing the missing transcript with the <star> token:
transcript = "*|THIS|MOMENT"
word_segments, frame_alignment = compute_and_plot_alignments(transcript, dictionary, emission, waveform)


######################################################################
# Conclusion
# ----------
#
# In this tutorial, we looked at how to use torchaudio’s forced alignment
# API to align and segment speech files, and demonstrated one advanced usage:
# How introducing a <star> token could improve alignment accuracy when
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
