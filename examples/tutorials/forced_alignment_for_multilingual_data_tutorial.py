"""
Forced alignment for multilingual data
======================================

**Author**: `Xiaohui Zhang <xiaohuizhang@meta.com>`__

This tutorial shows how to compute forced alignments for speech data
from multiple non-English languages using ``torchaudio``'s CTC forced alignment
API described in `“CTC forced alignment
tutorial” <https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html>`__
and the multilingual Wav2vec2 model proposed in the paper `“Scaling
Speech Technology to 1,000+
Languages” <https://research.facebook.com/publications/scaling-speech-technology-to-1000-languages/>`__.
The model was trained on 23K of audio data from 1100+ languages using
the `“uroman vocabulary” <https://www.isi.edu/~ulf/uroman.html>`__
as targets.

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
# Preparation
# --------------
#
# Here we import necessary packages, and define utility functions for
# computing the frame-level alignments (using the API
# ``functional.forced_align``), token-level and word-level alignments, and
# also alignment visualization utilities.
#

# %matplotlib inline
from dataclasses import dataclass

import IPython

import matplotlib.pyplot as plt

torch.random.manual_seed(0)

sample_rate = 16000


@dataclass
class Frame:
    # This is the index of each token in the transcript,
    # i.e. the current frame aligns to the N-th character from the transcript.
    token_index: int
    time_index: int
    score: float


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


# compute frame-level and word-level alignments using torchaudio's forced alignment API
def compute_alignments(transcript, dictionary, emission):
    frames = []
    tokens = [dictionary[c] for c in transcript.replace(" ", "")]

    targets = torch.tensor(tokens, dtype=torch.int32)
    input_lengths = torch.tensor(emission.shape[0])
    target_lengths = torch.tensor(targets.shape[0])

    # This is the key step, where we call the forced alignment API functional.forced_align to compute frame alignments.
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

    # compute frame alignments from token alignments
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

    # compue word alignments from token alignments
    separator = " "
    words = []
    i1, i2, i3 = 0, 0, 0
    while i3 < len(transcript):
        if i3 == len(transcript) - 1 or transcript[i3] == separator:
            if i1 != i2:
                if i3 == len(transcript) - 1:
                    i2 += 1
                s = 0
                segs = segments[i1 + s : i2 + s]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1 + s].start, segments[i2 + s - 1].end, score))
            i1 = i2
        else:
            i2 += 1
        i3 += 1

    num_frames = len(frame_alignment)
    return segments, words, num_frames


# utility function for plotting word alignments
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


# utility function for playing audio segments.
# A trick to embed the resulting audio to the generated file.
# `IPython.display.Audio` has to be the last call in a cell,
# and there should be only one call par cell.
def display_segment(i, waveform, word_segments, num_frames):
    ratio = waveform.size(1) / num_frames
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


######################################################################
# Aligning non-English data
# -----------------------------
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


def get_emission(waveform):
    # NOTE: this step is essential
    waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)

    emissions, _ = model(waveform)
    emissions = torch.log_softmax(emissions, dim=-1)
    emission = emissions[0].cpu().detach()

    # Append the extra dimension corresponding to the <star> token
    extra_dim = torch.zeros(emissions.shape[0], emissions.shape[1], 1)
    emissions = torch.cat((emissions.cpu(), extra_dim), 2)
    emission = emissions[0].detach()
    return emission, waveform


# Construct the dictionary
# '@' represents the OOV token, '*' represents the <star> token.
# <pad> and </s> are fairseq's legacy tokens, which're not used.
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
    "*": 31,
}


######################################################################
# Before aligning the speech with transcripts, we need to make sure
# the transcripts are already romanized. Here are the BASH commands
# required for saving raw transcript to a file, downloading the uroman
# romanizer and using it to obtain romanized transcripts, and PyThon
# commands required for further normalizing the romanized transcript.
#

# %%
# .. code-block:: bash
#
#    %%bash
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
# German example:
# ~~~~~~~~~~~~~~~~

text_raw = (
    "aber seit ich bei ihnen das brot hole brauch ich viel weniger schulze wandte sich ab die kinder taten ihm leid"
)
text_normalized = (
    "aber seit ich bei ihnen das brot hole brauch ich viel weniger schulze wandte sich ab die kinder taten ihm leid"
)
SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/10349_8674_000087.flac")
waveform, _ = torchaudio.load(SPEECH_FILE)

emission, waveform = get_emission(waveform)
assert len(dictionary) == emission.shape[1]

transcript = text_normalized

segments, word_segments, num_frames = compute_alignments(transcript, dictionary, emission)
plot_alignments(segments, word_segments, waveform[0], emission.shape[0])

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

display_segment(0, waveform, word_segments, num_frames)


######################################################################
#

display_segment(1, waveform, word_segments, num_frames)

######################################################################
#

display_segment(2, waveform, word_segments, num_frames)

######################################################################
#

display_segment(3, waveform, word_segments, num_frames)


######################################################################
#

display_segment(4, waveform, word_segments, num_frames)

######################################################################
#

display_segment(5, waveform, word_segments, num_frames)

######################################################################
#

display_segment(6, waveform, word_segments, num_frames)

######################################################################
#

display_segment(7, waveform, word_segments, num_frames)

######################################################################
#

display_segment(8, waveform, word_segments, num_frames)

######################################################################
#

display_segment(9, waveform, word_segments, num_frames)

######################################################################
#

display_segment(10, waveform, word_segments, num_frames)

######################################################################
#

display_segment(11, waveform, word_segments, num_frames)

######################################################################
#

display_segment(12, waveform, word_segments, num_frames)

######################################################################
#

display_segment(13, waveform, word_segments, num_frames)

######################################################################
#

display_segment(14, waveform, word_segments, num_frames)

######################################################################
#

display_segment(15, waveform, word_segments, num_frames)

######################################################################
#

display_segment(16, waveform, word_segments, num_frames)

######################################################################
#

display_segment(17, waveform, word_segments, num_frames)

######################################################################
#

display_segment(18, waveform, word_segments, num_frames)

######################################################################
#

display_segment(19, waveform, word_segments, num_frames)

######################################################################
#

display_segment(20, waveform, word_segments, num_frames)


######################################################################
# Chinese example:
# ~~~~~~~~~~~~~~~~
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
SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/mvdr/clean_speech.wav")
waveform, _ = torchaudio.load(SPEECH_FILE)

emission, waveform = get_emission(waveform)

transcript = text_normalized

segments, word_segments, num_frames = compute_alignments(transcript, dictionary, emission)
plot_alignments(segments, word_segments, waveform[0], emission.shape[0])

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

display_segment(0, waveform, word_segments, num_frames)


######################################################################
#

display_segment(1, waveform, word_segments, num_frames)

######################################################################
#

display_segment(2, waveform, word_segments, num_frames)

######################################################################
#

display_segment(3, waveform, word_segments, num_frames)


######################################################################
#

display_segment(4, waveform, word_segments, num_frames)

######################################################################
#

display_segment(5, waveform, word_segments, num_frames)

######################################################################
#

display_segment(6, waveform, word_segments, num_frames)

######################################################################
#

display_segment(7, waveform, word_segments, num_frames)

######################################################################
#

display_segment(8, waveform, word_segments, num_frames)


######################################################################
# Polish example:
# ~~~~~~~~~~~~~~~


text_raw = "wtedy ujrzałem na jego brzuchu okrągłą czarną ranę dlaczego mi nie powiedziałeś szepnąłem ze łzami"
text_normalized = "wtedy ujrzalem na jego brzuchu okragla czarna rane dlaczego mi nie powiedziales szepnalem ze lzami"
SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/5090_1447_000088.flac")
waveform, _ = torchaudio.load(SPEECH_FILE)

emission, waveform = get_emission(waveform)

transcript = text_normalized

segments, word_segments, num_frames = compute_alignments(transcript, dictionary, emission)
plot_alignments(segments, word_segments, waveform[0], emission.shape[0])

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

display_segment(0, waveform, word_segments, num_frames)


######################################################################
#

display_segment(1, waveform, word_segments, num_frames)

######################################################################
#

display_segment(2, waveform, word_segments, num_frames)

######################################################################
#

display_segment(3, waveform, word_segments, num_frames)


######################################################################
#

display_segment(4, waveform, word_segments, num_frames)

######################################################################
#

display_segment(5, waveform, word_segments, num_frames)

######################################################################
#

display_segment(6, waveform, word_segments, num_frames)

######################################################################
#

display_segment(7, waveform, word_segments, num_frames)

######################################################################
#

display_segment(8, waveform, word_segments, num_frames)

######################################################################
#

display_segment(9, waveform, word_segments, num_frames)

######################################################################
#

display_segment(10, waveform, word_segments, num_frames)

######################################################################
#

display_segment(11, waveform, word_segments, num_frames)

######################################################################
#

display_segment(12, waveform, word_segments, num_frames)

######################################################################
#

display_segment(13, waveform, word_segments, num_frames)

######################################################################
#

display_segment(14, waveform, word_segments, num_frames)


######################################################################
# Portuguese example:
# ~~~~~~~~~~~~~~~~~~~


text_raw = (
    "mas na imensa extensão onde se esconde o inconsciente imortal só me responde um bramido um queixume e nada mais"
)
text_normalized = (
    "mas na imensa extensao onde se esconde o inconsciente imortal so me responde um bramido um queixume e nada mais"
)
SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/6566_5323_000027.flac")
waveform, _ = torchaudio.load(SPEECH_FILE)

emission, waveform = get_emission(waveform)

transcript = text_normalized

segments, word_segments, num_frames = compute_alignments(transcript, dictionary, emission)
plot_alignments(segments, word_segments, waveform[0], emission.shape[0])

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

display_segment(0, waveform, word_segments, num_frames)


######################################################################
#

display_segment(1, waveform, word_segments, num_frames)

######################################################################
#

display_segment(2, waveform, word_segments, num_frames)

######################################################################
#

display_segment(3, waveform, word_segments, num_frames)


######################################################################
#

display_segment(4, waveform, word_segments, num_frames)

######################################################################
#

display_segment(5, waveform, word_segments, num_frames)

######################################################################
#

display_segment(6, waveform, word_segments, num_frames)

######################################################################
#

display_segment(7, waveform, word_segments, num_frames)

######################################################################
#

display_segment(8, waveform, word_segments, num_frames)

######################################################################
#

display_segment(9, waveform, word_segments, num_frames)

######################################################################
#

display_segment(10, waveform, word_segments, num_frames)

######################################################################
#

display_segment(11, waveform, word_segments, num_frames)

######################################################################
#

display_segment(12, waveform, word_segments, num_frames)

######################################################################
#

display_segment(13, waveform, word_segments, num_frames)

######################################################################
#

display_segment(14, waveform, word_segments, num_frames)

######################################################################
#

display_segment(15, waveform, word_segments, num_frames)

######################################################################
#

display_segment(16, waveform, word_segments, num_frames)

######################################################################
#

display_segment(17, waveform, word_segments, num_frames)

######################################################################
#

display_segment(18, waveform, word_segments, num_frames)

######################################################################
#

display_segment(19, waveform, word_segments, num_frames)


######################################################################
# Italian example:
# ~~~~~~~~~~~~~~~~

text_raw = "elle giacean per terra tutte quante fuor d'una ch'a seder si levò ratto ch'ella ci vide passarsi davante"
text_normalized = (
    "elle giacean per terra tutte quante fuor d'una ch'a seder si levo ratto ch'ella ci vide passarsi davante"
)
SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/642_529_000025.flac")
waveform, _ = torchaudio.load(SPEECH_FILE)

emission, waveform = get_emission(waveform)

transcript = text_normalized

segments, word_segments, num_frames = compute_alignments(transcript, dictionary, emission)
plot_alignments(segments, word_segments, waveform[0], emission.shape[0])

print("Raw Transcript: ", text_raw)
print("Normalized Transcript: ", text_normalized)
IPython.display.Audio(waveform, rate=sample_rate)

######################################################################
#

display_segment(0, waveform, word_segments, num_frames)


######################################################################
#

display_segment(1, waveform, word_segments, num_frames)

######################################################################
#

display_segment(2, waveform, word_segments, num_frames)

######################################################################
#

display_segment(3, waveform, word_segments, num_frames)


######################################################################
#

display_segment(4, waveform, word_segments, num_frames)

######################################################################
#

display_segment(5, waveform, word_segments, num_frames)

######################################################################
#

display_segment(6, waveform, word_segments, num_frames)

######################################################################
#

display_segment(7, waveform, word_segments, num_frames)

######################################################################
#

display_segment(8, waveform, word_segments, num_frames)

######################################################################
#

display_segment(9, waveform, word_segments, num_frames)

######################################################################
#

display_segment(10, waveform, word_segments, num_frames)

######################################################################
#

display_segment(11, waveform, word_segments, num_frames)

######################################################################
#

display_segment(12, waveform, word_segments, num_frames)

######################################################################
#

display_segment(13, waveform, word_segments, num_frames)

######################################################################
#

display_segment(14, waveform, word_segments, num_frames)

######################################################################
#

display_segment(15, waveform, word_segments, num_frames)

######################################################################
#

display_segment(16, waveform, word_segments, num_frames)

######################################################################
#

display_segment(17, waveform, word_segments, num_frames)


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
