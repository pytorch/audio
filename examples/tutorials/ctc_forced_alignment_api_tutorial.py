"""
CTC forced alignment API tutorial
=================================

**Author**: `Xiaohui Zhang <xiaohuizhang@meta.com>`__, `Moto Hira <moto@meta.com>`__

The forced alignment is a process to align transcript with speech.
This tutorial shows how to align transcripts to speech using
:py:func:`torchaudio.functional.forced_align` which was developed along the work of
`Scaling Speech Technology to 1,000+ Languages
<https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/>`__.

:py:func:`~torchaudio.functional.forced_align` has custom CPU and CUDA
implementations which are more performant than the vanilla Python
implementation above, and are more accurate.
It can also handle missing transcript with special ``<star>`` token.

There is also a high-level API, :py:class:`torchaudio.pipelines.Wav2Vec2FABundle`,
which wraps the pre/post-processing explained in this tutorial and makes it easy
to run forced-alignments.
`Forced alignment for multilingual data
<./forced_alignment_for_multilingual_data_tutorial.html>`__ uses this API to
illustrate how to align non-English transcripts.
"""

######################################################################
# Preparation
# -----------

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

######################################################################
#

import IPython
import matplotlib.pyplot as plt

import torchaudio.functional as F

######################################################################
# First we prepare the speech data and the transcript we area going
# to use.
#

SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
waveform, _ = torchaudio.load(SPEECH_FILE)
TRANSCRIPT = "i had that curiosity beside me at this moment".split()


######################################################################
# Generating emissions
# ~~~~~~~~~~~~~~~~~~~~
#
# :py:func:`~torchaudio.functional.forced_align` takes emission and
# token sequences and outputs timestaps of the tokens and their scores.
#
# Emission reperesents the frame-wise probability distribution over
# tokens, and it can be obtained by passing waveform to an acoustic
# model.
#
# Tokens are numerical expression of transcripts. There are many ways to
# tokenize transcripts, but here, we simply map alphabets into integer,
# which is how labels were constructed when the acoustice model we are
# going to use was trained.
#
# We will use a pre-trained Wav2Vec2 model,
# :py:data:`torchaudio.pipelines.MMS_FA`, to obtain emission and tokenize
# the transcript.
#

bundle = torchaudio.pipelines.MMS_FA

model = bundle.get_model(with_star=False).to(device)
with torch.inference_mode():
    emission, _ = model(waveform.to(device))


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
# Tokenize the transcript
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# We create a dictionary, which maps each label into token.

LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)
for k, v in DICTIONARY.items():
    print(f"{k}: {v}")

######################################################################
# converting transcript to tokens is as simple as

tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPT for c in word]

for t in tokenized_transcript:
    print(t, end=" ")
print()

######################################################################
# Computing frame-level alignments
# --------------------------------
#
# Now we call TorchAudio’s forced alignment API to compute the
# frame-level alignment. For the detail of function signature, please
# refer to :py:func:`~torchaudio.functional.forced_align`.
#


def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

######################################################################
# Now let's look at the output.

for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
    print(f"{i:3d}:\t{ali:2d} [{LABELS[ali]}], {score:.2f}")

######################################################################
#
# .. note::
#
#    The alignment is expressed in the frame cordinate of the emission,
#    which is different from the original waveform.
#
# It contains blank tokens and repeated tokens. The following is the
# interpretation of the non-blank tokens.
#
# .. code-block::
#
#    31:     0 [-], 1.00
#    32:     2 [i], 1.00  "i" starts and ends
#    33:     0 [-], 1.00
#    34:     0 [-], 1.00
#    35:    15 [h], 1.00  "h" starts
#    36:    15 [h], 0.93  "h" ends
#    37:     1 [a], 1.00  "a" starts and ends
#    38:     0 [-], 0.96
#    39:     0 [-], 1.00
#    40:     0 [-], 1.00
#    41:    13 [d], 1.00  "d" starts and ends
#    42:     0 [-], 1.00
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

######################################################################
# Obtain token-level alignment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next step is to resolve the repetation, so that each alignment does
# not depend on previous alignments.
# :py:func:`torchaudio.functional.merge_tokens` computes the
# :py:class:`~torchaudio.functional.TokenSpan` object, which represents
# which token from the transcript is present at what time span.

######################################################################
#

token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

print("Token\tTime\tScore")
for s in token_spans:
    print(f"{LABELS[s.token]}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")


######################################################################
# Visualization
# ~~~~~~~~~~~~~
#
def plot_scores(spans, scores):
    fig, ax = plt.subplots()
    ax.set_title("frame-level and token-level confidence scores")
    span_xs, span_hs, span_ws = [], [], []
    frame_xs, frame_hs = [], []
    for span in spans:
        if LABELS[span.token] != "|":
            span_xs.append((span.end + span.start) / 2 + 0.4)
            span_hs.append(span.score)
            span_ws.append(span.end - span.start)
            ax.annotate(LABELS[span.token], (span.start + 0.8, -0.07), weight="bold")
            for t in range(span.start, span.end):
                frame_xs.append(t + 1)
                frame_hs.append(scores[t].item())
    ax.bar(span_xs, span_hs, width=span_ws, color="gray", alpha=0.5, edgecolor="black")
    ax.bar(frame_xs, frame_hs, width=0.5, alpha=0.5)

    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, axis="y")
    fig.tight_layout()


plot_scores(token_spans, alignment_scores)


######################################################################
# Obtain word-level alignments and confidence scores
# --------------------------------------------------
#
# Now let’s merge the token-level alignments and confidence scores to get
# word-level alignments and confidence scores. Then, finally, we verify
# the quality of word alignments by 1) plotting the word-level alignments
# and the waveform, 2) segmenting the original audio according to the
# alignments and listening to them.


# Obtain word alignments from token alignments
def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPT])


######################################################################
# Visualization
# ~~~~~~~~~~~~~
#

# Compute average score weighted by the span length
def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def plot_alignments(waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start + 0.1, t_spans[-1].end - 0.1
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    axes[1].set_xlim([0, None])
    fig.tight_layout()


plot_alignments(waveform, word_spans, emission, TRANSCRIPT)


######################################################################
def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


num_frames = emission.size(1)

######################################################################

# Generate the audio for each segment
print(TRANSCRIPT)
IPython.display.Audio(SPEECH_FILE)

######################################################################
#

preview_word(waveform, word_spans[0], num_frames, TRANSCRIPT[0])

######################################################################
#

preview_word(waveform, word_spans[1], num_frames, TRANSCRIPT[1])

######################################################################
#

preview_word(waveform, word_spans[2], num_frames, TRANSCRIPT[2])

######################################################################
#

preview_word(waveform, word_spans[3], num_frames, TRANSCRIPT[3])

######################################################################
#

preview_word(waveform, word_spans[4], num_frames, TRANSCRIPT[4])

######################################################################
#

preview_word(waveform, word_spans[5], num_frames, TRANSCRIPT[5])

######################################################################
#

preview_word(waveform, word_spans[6], num_frames, TRANSCRIPT[6])

######################################################################
#

preview_word(waveform, word_spans[7], num_frames, TRANSCRIPT[7])

######################################################################
#

preview_word(waveform, word_spans[8], num_frames, TRANSCRIPT[8])


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

star_dim = torch.zeros((1, emission.size(1), 1), device=emission.device, dtype=emission.dtype)
emission = torch.cat((emission, star_dim), 2)

assert len(DICTIONARY) == emission.shape[2]

plot_emission(emission[0])

######################################################################
# The following function combines all the processes, and compute
# word segments from emission in one-go.


def compute_alignments(emission, transcript, dictionary):
    tokens = [dictionary[char] for word in transcript for char in word]
    alignment, scores = align(emission, tokens)
    token_spans = F.merge_tokens(alignment, scores)
    word_spans = unflatten(token_spans, [len(word) for word in transcript])
    return word_spans


######################################################################
# **Original**

word_spans = compute_alignments(emission, TRANSCRIPT, DICTIONARY)
plot_alignments(waveform, word_spans, emission, TRANSCRIPT)

######################################################################
# **With <star> token**
#
# Now we replace the first part of the transcript with the ``<star>`` token.

transcript = "* this moment".split()
word_spans = compute_alignments(emission, transcript, DICTIONARY)
plot_alignments(waveform, word_spans, emission, transcript)

######################################################################
#

preview_word(waveform, word_spans[0], num_frames, transcript[0])

######################################################################
#

preview_word(waveform, word_spans[1], num_frames, transcript[1])

######################################################################
#

preview_word(waveform, word_spans[2], num_frames, transcript[2])

######################################################################
#

######################################################################
# **Without <star> token**
#
# As a comparison, the following aligns the partial transcript
# without using ``<star>`` token.
# It demonstrates the effect of ``<star>`` token for dealing with deletion errors.

transcript = "this moment".split()
word_spans = compute_alignments(emission, transcript, DICTIONARY)
plot_alignments(waveform, word_spans, emission, transcript)

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
# Ni <zni@meta.com>`__ for developing and open-sourcing the
# forced aligner API.
#
