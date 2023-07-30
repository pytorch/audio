"""
Forced alignment for multilingual data
======================================

**Authors**: `Xiaohui Zhang <xiaohuizhang@meta.com>`__, `Moto Hira <moto@meta.com>`__.

This tutorial shows how to align transcript to speech for non-English languages.

The process of aligning non-English (normalized) transcript is identical to aligning
English (normalized) transcript, and the process for English is covered in detail in
`CTC forced alignment tutorial <./ctc_forced_alignment_api_tutorial.html>`__.
In this tutorial, we use TorchAudio's high-level API,
:py:class:`torchaudio.pipelines.Wav2Vec2FABundle`, which packages the pre-trained
model, tokenizer and aligner, to perform the forced alignment with less code.
"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


######################################################################
#
from typing import List

import IPython
import matplotlib.pyplot as plt

######################################################################
# Creating the pipeline
# ---------------------
#
# First, we instantiate the model and pre/post-processing pipelines.
#
# The following diagram illustrates the process of alignment.
#
# .. image:: https://download.pytorch.org/torchaudio/doc-assets/pipelines-wav2vec2fabundle.png
#
# The waveform is passed to an acoustic model, which produces the sequence of
# probability distribution of tokens.
# The transcript is passed to tokenizer, which converts the transcript to
# sequence of tokens.
# Aligner takes the results from the acoustic model and the tokenizer and generate
# timestamps for each token.
#
# .. note::
#
#    This process expects that the input transcript is already normalized.
#    The process of normalization, which involves romanization of non-English
#    languages, is language-dependent, so it is not covered in this tutorial,
#    but we will breifly look into it.
#
# The acoustic model and the tokenizer must use the same set of tokens.
# To facilitate the creation of matching processors,
# :py:class:`~torchaudio.pipelines.Wav2Vec2FABundle` associates a
# pre-trained accoustic model and a tokenizer.
# :py:data:`torchaudio.pipelines.MMS_FA` is one of such instance.
#
# The following code instantiates a pre-trained acoustic model, a tokenizer
# which uses the same set of tokens as the model, and an aligner.
#
from torchaudio.pipelines import MMS_FA as bundle

model = bundle.get_model()
model.to(device)

tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()


######################################################################
# .. note::
#
#    The model instantiated by :py:data:`~torchaudio.pipelines.MMS_FA`'s
#    :py:meth:`~torchaudio.pipelines.Wav2Vec2FABundle.get_model`
#    method by default includes the feature dimension for ``<star>`` token.
#    You can disable this by passing ``with_star=False``.
#

######################################################################
# The acoustic model of :py:data:`~torchaudio.pipelines.MMS_FA` was
# created and open-sourced as part of the research project,
# `Scaling Speech Technology to 1,000+ Languages
# <https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/>`__.
# It was trained with 23,000 hours of audio from 1100+ languages.
#
# The tokenizer simply maps the normalized characters to integers.
# You can check the mapping as follow;

print(bundle.get_dict())


######################################################################
#
# The aligner internally uses :py:func:`torchaudio.functional.forced_align`
# and :py:func:`torchaudio.functional.merge_tokens` to infer the time
# stamps of the input tokens.
#
# The detail of the underlying mechanism is covered in
# `CTC forced alignment API tutorial <./ctc_forced_alignment_api_tutorial.html>`__,
# so please refer to it.

######################################################################
# We define a utility function that performs the forced alignment with
# the above model, the tokenizer and the aligner.
#
def compute_alignments(waveform: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
        token_spans = aligner(emission[0], tokenizer(transcript))
    return emission, token_spans


######################################################################
# We also define utility functions for plotting the result and previewing
# the audio segments.

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
        t0, t1 = t_spans[0].start, t_spans[-1].end
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    fig.tight_layout()


######################################################################
#
def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


######################################################################
# Normalizing the transcript
# --------------------------
#
# The transcripts passed to the pipeline must be normalized beforehand.
# The exact process of normalization depends on language.
#
# Languages that do not have explicit word boundaries
# (such as Chinese, Japanese and Korean) require segmentation first.
# There are dedicated tools for this, but let's say we have segmented
# transcript.
#
# The first step of normalization is romanization.
# `uroman <https://github.com/isi-nlp/uroman>`__ is a tool that
# supports many languages.
#
# Here is a BASH commands to romanize the input text file and write
# the output to another text file using ``uroman``.
#
# .. code-block:: bash
#
#    $ echo "des événements d'actualité qui se sont produits durant l'année 1882" > text.txt
#    $ uroman/bin/uroman.pl < text.txt > text_romanized.txt
#    $ cat text_romanized.txt
#
# .. code-block:: text
#
#    Cette page concerne des evenements d'actualite qui se sont produits durant l'annee 1882
#
# The next step is to remove non-alphabets and punctuations.
# The following snippet normalizes the romanized transcript.
#
# .. code-block:: python
#
#    import re
#
#
#    def normalize_uroman(text):
#        text = text.lower()
#        text = text.replace("’", "'")
#        text = re.sub("([^a-z' ])", " ", text)
#        text = re.sub(' +', ' ', text)
#        return text.strip()
#
#
#    with open("text_romanized.txt", "r") as f:
#        for line in f:
#            text_normalized = normalize_uroman(line)
#            print(text_normalized)
#
# Running the script on the above exanple produces the following.
#
# .. code-block:: text
#
#    cette page concerne des evenements d'actualite qui se sont produits durant l'annee
#
# Note that, in this example, since "1882" was not romanized by ``uroman``,
# it was removed in the normalization step.
# To avoid this, one needs to romanize numbers, but this is known to be a non-trivial task.
#

######################################################################
# Aligning transcripts to speech
# ------------------------------
#
# Now we perform the forced alignment for multiple languages.
#
#
# German
# ~~~~~~

text_raw = "aber seit ich bei ihnen das brot hole"
text_normalized = "aber seit ich bei ihnen das brot hole"

url = "https://download.pytorch.org/torchaudio/tutorial-assets/10349_8674_000087.flac"
waveform, sample_rate = torchaudio.load(
    url, frame_offset=int(0.5 * bundle.sample_rate), num_frames=int(2.5 * bundle.sample_rate)
)

######################################################################
#
assert sample_rate == bundle.sample_rate

######################################################################
#

transcript = text_normalized.split()
tokens = tokenizer(transcript)

emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)

plot_alignments(waveform, token_spans, emission, transcript)

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

preview_word(waveform, token_spans[0], num_frames, transcript[0])

######################################################################
#

preview_word(waveform, token_spans[1], num_frames, transcript[1])

######################################################################
#

preview_word(waveform, token_spans[2], num_frames, transcript[2])

######################################################################
#

preview_word(waveform, token_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, token_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, token_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, token_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, token_spans[7], num_frames, transcript[7])

######################################################################
# Chinese
# ~~~~~~~
#
# Chinese is a character-based language, and there is not explicit word-level
# tokenization (separated by spaces) in its raw written form. In order to
# obtain word level alignments, you need to first tokenize the transcripts
# at the word level using a word tokenizer like `“Stanford
# Tokenizer” <https://michelleful.github.io/code-blog/2015/09/10/parsing-chinese-with-stanford/>`__.
# However this is not needed if you only want character-level alignments.
#

text_raw = "关 服务 高端 产品 仍 处于 供不应求 的 局面"
text_normalized = "guan fuwu gaoduan chanpin reng chuyu gongbuyingqiu de jumian"

######################################################################
#

url = "https://download.pytorch.org/torchaudio/tutorial-assets/mvdr/clean_speech.wav"
waveform, sample_rate = torchaudio.load(url)
waveform = waveform[0:1]

######################################################################
#
assert sample_rate == bundle.sample_rate

######################################################################
#

transcript = text_normalized.split()
emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)

plot_alignments(waveform, token_spans, emission, transcript)

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

preview_word(waveform, token_spans[0], num_frames, transcript[0])

######################################################################
#

preview_word(waveform, token_spans[1], num_frames, transcript[1])

######################################################################
#

preview_word(waveform, token_spans[2], num_frames, transcript[2])

######################################################################
#

preview_word(waveform, token_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, token_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, token_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, token_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, token_spans[7], num_frames, transcript[7])

######################################################################
#

preview_word(waveform, token_spans[8], num_frames, transcript[8])


######################################################################
# Polish
# ~~~~~~

text_raw = "wtedy ujrzałem na jego brzuchu okrągłą czarną ranę"
text_normalized = "wtedy ujrzalem na jego brzuchu okragla czarna rane"

url = "https://download.pytorch.org/torchaudio/tutorial-assets/5090_1447_000088.flac"
waveform, sample_rate = torchaudio.load(url, num_frames=int(4.5 * bundle.sample_rate))

######################################################################
#
assert sample_rate == bundle.sample_rate

######################################################################
#

transcript = text_normalized.split()
emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)

plot_alignments(waveform, token_spans, emission, transcript)

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

preview_word(waveform, token_spans[0], num_frames, transcript[0])

######################################################################
#

preview_word(waveform, token_spans[1], num_frames, transcript[1])

######################################################################
#

preview_word(waveform, token_spans[2], num_frames, transcript[2])

######################################################################
#

preview_word(waveform, token_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, token_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, token_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, token_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, token_spans[7], num_frames, transcript[7])

######################################################################
# Portuguese
# ~~~~~~~~~~

text_raw = "na imensa extensão onde se esconde o inconsciente imortal"
text_normalized = "na imensa extensao onde se esconde o inconsciente imortal"

url = "https://download.pytorch.org/torchaudio/tutorial-assets/6566_5323_000027.flac"
waveform, sample_rate = torchaudio.load(
    url, frame_offset=int(bundle.sample_rate), num_frames=int(4.6 * bundle.sample_rate)
)

######################################################################
#
assert sample_rate == bundle.sample_rate

######################################################################
#

transcript = text_normalized.split()
emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)

plot_alignments(waveform, token_spans, emission, transcript)

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

preview_word(waveform, token_spans[0], num_frames, transcript[0])

######################################################################
#

preview_word(waveform, token_spans[1], num_frames, transcript[1])

######################################################################
#

preview_word(waveform, token_spans[2], num_frames, transcript[2])

######################################################################
#

preview_word(waveform, token_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, token_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, token_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, token_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, token_spans[7], num_frames, transcript[7])

######################################################################
#

preview_word(waveform, token_spans[8], num_frames, transcript[8])


######################################################################
# Italian
# ~~~~~~~

text_raw = "elle giacean per terra tutte quante"
text_normalized = "elle giacean per terra tutte quante"

url = "https://download.pytorch.org/torchaudio/tutorial-assets/642_529_000025.flac"
waveform, sample_rate = torchaudio.load(url, num_frames=int(4 * bundle.sample_rate))

######################################################################
#
assert sample_rate == bundle.sample_rate

######################################################################
#

transcript = text_normalized.split()
emission, token_spans = compute_alignments(waveform, transcript)
num_frames = emission.size(1)

plot_alignments(waveform, token_spans, emission, transcript)

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

preview_word(waveform, token_spans[0], num_frames, transcript[0])

######################################################################
#

preview_word(waveform, token_spans[1], num_frames, transcript[1])

######################################################################
#

preview_word(waveform, token_spans[2], num_frames, transcript[2])

######################################################################
#

preview_word(waveform, token_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, token_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, token_spans[5], num_frames, transcript[5])

######################################################################
# Conclusion
# ----------
#
# In this tutorial, we looked at how to use torchaudio’s forced alignment
# API and a Wav2Vec2 pre-trained mulilingual acoustic model to align
# speech data to transcripts in five languages.
#

######################################################################
# Acknowledgement
# ---------------
#
# Thanks to `Vineel Pratap <vineelkpratap@meta.com>`__ and `Zhaoheng
# Ni <zni@meta.com>`__ for developing and open-sourcing the
# forced aligner API.
#
