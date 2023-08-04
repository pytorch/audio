"""
Forced alignment for multilingual data
======================================

**Author**: `Xiaohui Zhang <xiaohuizhang@meta.com>`__

This tutorial shows how to compute forced alignments for speech data
from multiple non-English languages using ``torchaudio``'s CTC forced alignment
API described in `CTC forced alignment tutorial <./forced_alignment_tutorial.html>`__
and the multilingual Wav2vec2 model proposed in the paper `Scaling
Speech Technology to 1,000+
Languages <https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/>`__.

The model was trained on 23K of audio data from 1100+ languages using
the `uroman vocabulary <https://www.isi.edu/~ulf/uroman.html>`__
as targets.
"""

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from dataclasses import dataclass

######################################################################
# Preparation
# -----------
#
from typing import Dict, List

import IPython
import matplotlib.pyplot as plt
from torchaudio.functional import forced_align


######################################################################
#

SAMPLE_RATE = 16000


######################################################################
#
# Here we define utility functions for computing the frame-level
# alignments (using the API :py:func:`torchaudio.functional.forced_align`),
# token-level and word-level alignments.
# For the detail of these functions please refer to
# `CTC forced alignment API tutorial <./ctc_forced_alignment_api_tutorial.html>`__.
#


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


@dataclass
class WordSpan:
    token_spans: List[TokenSpan]
    score: float


######################################################################
#
def align_emission_and_tokens(emission: torch.Tensor, tokens: List[int]):
    device = emission.device
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    input_lengths = torch.tensor([emission.size(1)], device=device)
    target_lengths = torch.tensor([targets.size(1)], device=device)

    aligned_tokens, scores = forced_align(emission, targets, input_lengths, target_lengths, 0)

    scores = scores.exp()  # convert back to probability
    aligned_tokens, scores = aligned_tokens[0], scores[0]  # remove batch dimension
    return aligned_tokens, scores


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


def merge_words(token_spans: List[TokenSpan], transcript: List[str]) -> List[WordSpan]:
    def _score(t_spans):
        return sum(s.score * len(s) for s in t_spans) / sum(len(s) for s in t_spans)

    word_spans = []
    i = 0
    for words in transcript:
        j = i + len(words)
        word_spans.append(WordSpan(token_spans[i:j], _score(token_spans[i:j])))
        i = j
    return word_spans


def compute_alignments(emission: torch.Tensor, transcript: List[str], dictionary: Dict[str, int]):
    tokens = [dictionary[c] for word in transcript for c in word]
    aligned_tokens, scores = align_emission_and_tokens(emission, tokens)
    token_spans = merge_tokens(aligned_tokens, scores)
    word_spans = merge_words(token_spans, transcript)
    return word_spans


######################################################################
#

# utility function for plotting word alignments
def plot_alignments(waveform, word_spans, emission, transcript, sample_rate=SAMPLE_RATE):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for w_span, chars in zip(word_spans, transcript):
        t_spans = w_span.token_spans
        t0, t1 = t_spans[0].start, t_spans[-1].end
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{w_span.score:.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            axes[1].annotate(char, (span.start * ratio, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    fig.tight_layout()
    return IPython.display.Audio(waveform, rate=sample_rate)


######################################################################
#


def preview_word(waveform, word_span, num_frames, transcript, sample_rate=SAMPLE_RATE):
    ratio = waveform.size(1) / num_frames
    t0 = word_span.token_spans[0].start
    t1 = word_span.token_spans[-1].end
    x0 = int(ratio * t0)
    x1 = int(ratio * t1)
    print(f"{transcript} ({word_span.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


######################################################################
# Aligning multilingual data
# --------------------------
#
# Here we show examples of computing forced alignments of utterances in
# 5 languages using the multilingual Wav2vec2 model, with the alignments visualized.
# One can also play the whole audio and audio segments aligned with each word, in
# order to verify the alignment quality. Here we first load the model and dictionary.
#

from torchaudio.models import wav2vec2_model

model = wav2vec2_model(
    extractor_mode="layer_norm",
    extractor_conv_layer_config=[
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ],
    extractor_conv_bias=True,
    encoder_embed_dim=1024,
    encoder_projection_dropout=0.0,
    encoder_pos_conv_kernel=128,
    encoder_pos_conv_groups=16,
    encoder_num_layers=24,
    encoder_num_heads=16,
    encoder_attention_dropout=0.0,
    encoder_ff_interm_features=4096,
    encoder_ff_interm_dropout=0.1,
    encoder_dropout=0.0,
    encoder_layer_norm_first=True,
    encoder_layer_drop=0.1,
    aux_num_out=31,
)


model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
    )
)
model.eval()
model.to(device)


def get_emission(waveform):
    with torch.inference_mode():
        # NOTE: this step is essential
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
        emission, _ = model(waveform)
        return torch.log_softmax(emission, dim=-1)


# Construct the dictionary
# '@' represents the OOV token
# <pad> and </s> are fairseq's legacy tokens, which're not used.
# <star> token is omitted as we do not use it in this tutorial
dictionary = {
    "<blank>": 0,
    "<pad>": 1,
    "</s>": 2,
    "@": 3,
    "a": 4,
    "i": 5,
    "e": 6,
    "n": 7,
    "o": 8,
    "u": 9,
    "t": 10,
    "s": 11,
    "r": 12,
    "m": 13,
    "k": 14,
    "l": 15,
    "d": 16,
    "g": 17,
    "h": 18,
    "y": 19,
    "b": 20,
    "p": 21,
    "w": 22,
    "c": 23,
    "v": 24,
    "j": 25,
    "z": 26,
    "f": 27,
    "'": 28,
    "q": 29,
    "x": 30,
}


######################################################################
# Before aligning the speech with transcripts, we need to make sure
# the transcripts are already romanized. Here are the BASH commands
# required for saving raw transcript to a file, downloading the uroman
# romanizer and using it to obtain romanized transcripts, and PyThon
# commands required for further normalizing the romanized transcript.
#
# .. code-block:: bash
#
#    Save the raw transcript to a file
#    echo 'raw text' > text.txt
#    git clone https://github.com/isi-nlp/uroman
#    uroman/bin/uroman.pl < text.txt > text_romanized.txt
#

######################################################################
# .. code-block:: python
#
#    import re
#    def normalize_uroman(text):
#        text = text.lower()
#        text = text.replace("’", "'")
#        text = re.sub("([^a-z' ])", " ", text)
#        text = re.sub(' +', ' ', text)
#        return text.strip()
#
#    file = "text_romanized.txt"
#    f = open(file, "r")
#    lines = f.readlines()
#    text_normalized = normalize_uroman(lines[0].strip())
#


######################################################################
# German
# ~~~~~~

speech_file = torchaudio.utils.download_asset("tutorial-assets/10349_8674_000087.flac", progress=False)

text_raw = "aber seit ich bei ihnen das brot hole"
text_normalized = "aber seit ich bei ihnen das brot hole"

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)

######################################################################
#

waveform, _ = torchaudio.load(speech_file, frame_offset=int(0.5 * SAMPLE_RATE), num_frames=int(2.5 * SAMPLE_RATE))

emission = get_emission(waveform.to(device))
num_frames = emission.size(1)

######################################################################
#

transcript = text_normalized.split()
word_spans = compute_alignments(emission, transcript, dictionary)

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

preview_word(waveform, word_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, word_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, word_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, word_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, word_spans[7], num_frames, transcript[7])

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

speech_file = torchaudio.utils.download_asset("tutorial-assets/mvdr/clean_speech.wav", progress=False)

text_raw = "关 服务 高端 产品 仍 处于 供不应求 的 局面"
text_normalized = "guan fuwu gaoduan chanpin reng chuyu gongbuyingqiu de jumian"

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)

######################################################################
#

waveform, _ = torchaudio.load(speech_file)
waveform = waveform[0:1]

emission = get_emission(waveform.to(device))
num_frames = emission.size(1)

######################################################################
#

transcript = text_normalized.split()
word_spans = compute_alignments(emission, transcript, dictionary)

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

preview_word(waveform, word_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, word_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, word_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, word_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, word_spans[7], num_frames, transcript[7])

######################################################################
#

preview_word(waveform, word_spans[8], num_frames, transcript[8])


######################################################################
# Polish
# ~~~~~~

speech_file = torchaudio.utils.download_asset("tutorial-assets/5090_1447_000088.flac", progress=False)

text_raw = "wtedy ujrzałem na jego brzuchu okrągłą czarną ranę"
text_normalized = "wtedy ujrzalem na jego brzuchu okragla czarna rane"

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)

######################################################################
#

waveform, _ = torchaudio.load(speech_file, num_frames=int(4.5 * SAMPLE_RATE))

emission = get_emission(waveform.to(device))
num_frames = emission.size(1)

######################################################################
#

transcript = text_normalized.split()
word_spans = compute_alignments(emission, transcript, dictionary)

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

preview_word(waveform, word_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, word_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, word_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, word_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, word_spans[7], num_frames, transcript[7])

######################################################################
# Portuguese
# ~~~~~~~~~~

speech_file = torchaudio.utils.download_asset("tutorial-assets/6566_5323_000027.flac", progress=False)

text_raw = "na imensa extensão onde se esconde o inconsciente imortal"
text_normalized = "na imensa extensao onde se esconde o inconsciente imortal"

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)

######################################################################
#

waveform, _ = torchaudio.load(speech_file, frame_offset=int(SAMPLE_RATE), num_frames=int(4.6 * SAMPLE_RATE))

emission = get_emission(waveform.to(device))
num_frames = emission.size(1)

######################################################################
#

transcript = text_normalized.split()
word_spans = compute_alignments(emission, transcript, dictionary)

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

preview_word(waveform, word_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, word_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, word_spans[5], num_frames, transcript[5])

######################################################################
#

preview_word(waveform, word_spans[6], num_frames, transcript[6])

######################################################################
#

preview_word(waveform, word_spans[7], num_frames, transcript[7])

######################################################################
#

preview_word(waveform, word_spans[8], num_frames, transcript[8])


######################################################################
# Italian
# ~~~~~~~

speech_file = torchaudio.utils.download_asset("tutorial-assets/642_529_000025.flac", progress=False)

text_raw = "elle giacean per terra tutte quante"
text_normalized = "elle giacean per terra tutte quante"

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)

######################################################################
#

waveform, _ = torchaudio.load(speech_file, num_frames=int(4 * SAMPLE_RATE))

emission = get_emission(waveform.to(device))
num_frames = emission.size(1)

######################################################################
#

transcript = text_normalized.split()
word_spans = compute_alignments(emission, transcript, dictionary)

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

preview_word(waveform, word_spans[3], num_frames, transcript[3])

######################################################################
#

preview_word(waveform, word_spans[4], num_frames, transcript[4])

######################################################################
#

preview_word(waveform, word_spans[5], num_frames, transcript[5])

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
# Ni <zni@meta.com>`__ for working on the forced aligner API, and
# `Moto Hira <moto@meta.com>`__ for providing alignment merging and
# visualization utilities.
#
