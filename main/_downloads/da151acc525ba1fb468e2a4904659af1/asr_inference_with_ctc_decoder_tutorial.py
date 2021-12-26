"""
ASR Inference with CTC Decoder
==============================

**Author**: `Caroline Chen <carolinechen@fb.com>`__

This tutorial shows how to perform speech recognition inference using a
CTC beam search decoder with lexicon constraint and KenLM language model
support. We demonstrate this on a pretrained wav2vec 2.0 model trained
using CTC loss.

"""


######################################################################
# Overview
# --------
#
# Running ASR inference using a CTC Beam Search decoder with a KenLM
# language model and lexicon constraint requires the following components
#
# -  Acoustic Model: model predicting phonetics from audio waveforms
# -  Tokens: the possible predicted tokens from the acoustic model
# -  Lexicon: mapping between possible words and their corresponding
#    tokens sequence
# -  KenLM: n-gram language model trained with the `KenLM
#    library <https://kheafield.com/code/kenlm/>`__
#


######################################################################
# Preparation
# -----------
#
# First we import the necessary utilities and fetch the data that we are
# working with
#

import time

import IPython
import torch
import torchaudio


######################################################################
# Acoustic Model and Data
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# We use the pretrained `Wav2Vec 2.0 <https://arxiv.org/abs/2006.11477>`__
# Base model that is finetuned on 10 min of the `LibriSpeech
# dataset <http://www.openslr.org/12>`__, which can be loaded in using
# :py:func:`torchaudio.pipelines`. For more detail on running Wav2Vec 2.0 speech
# recognition pipelines in torchaudio, please refer to `this
# tutorial <https://pytorch.org/audio/main/tutorials/speech_recognition_pipeline_tutorial.html>`__.
#

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
acoustic_model = bundle.get_model()


######################################################################
# We will load a sample from the LibriSpeech test-other dataset.
#

hub_dir = torch.hub.get_dir()

speech_url = "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/1688-142285-0007.wav"
speech_file = f"{hub_dir}/speech.wav"

torch.hub.download_url_to_file(speech_url, speech_file)

IPython.display.Audio(speech_file)


######################################################################
# The transcript corresponding to this audio file is
# ::
#   i really was very much afraid of showing him how much shocked i was at some parts of what he said
#

waveform, sample_rate = torchaudio.load(speech_file)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


######################################################################
# Files and Data for Decoder
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we load in our token, lexicon, and KenLM data, which are used by
# the decoder to predict words from the acoustic model output.
#
# Note: this cell may take a couple of minutes to run, as the language
# model can be large
#


######################################################################
# Tokens
# ^^^^^^
#
# The tokens are the possible symbols that the acoustic model can predict,
# including the blank and silent symbols. It can either be passed in as a
# file, where each line consists of the tokens corresponding to the same
# index, or as a list of tokens, each mapping to a unique index.
#
# ::
#
#    # tokens.txt
#    _
#    |
#    e
#    t
#    ...
#

tokens = [label.lower() for label in bundle.get_labels()]
print(tokens)


######################################################################
# Lexicon
# ^^^^^^^
#
# The lexicon is a mapping from words to their corresponding tokens
# sequence, and is used to restrict the search space of the decoder to
# only words from the lexicon. The expected format of the lexicon file is
# a line per word, with a word followed by its space-split tokens.
#
# ::
#
#    # lexcion.txt
#    a a |
#    able a b l e |
#    about a b o u t |
#    ...
#    ...
#

lexicon_url = "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/lexicon-librispeech.txt"
lexicon_file = f"{hub_dir}/lexicon.txt"
torch.hub.download_url_to_file(lexicon_url, lexicon_file)


######################################################################
# KenLM
# ^^^^^
#
# This is an n-gram language model trained with the `KenLM
# library <https://kheafield.com/code/kenlm/>`__. Both the ``.arpa`` or
# the binarized ``.bin`` LM can be used, but the binary format is
# recommended for faster loading.
#
# The language model used in this tutorial is a 4-gram KenLM trained using
# `LibriSpeech <http://www.openslr.org/11>`__.
#

kenlm_url = "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/4-gram-librispeech.bin"
kenlm_file = f"{hub_dir}/kenlm.bin"
torch.hub.download_url_to_file(kenlm_url, kenlm_file)


######################################################################
# Construct Beam Search Decoder
# -----------------------------
#
# The decoder can be constructed using the factory function
# :py:func:`kenlm_lexicon_decoder <torchaudio.prototype.ctc_decoder.kenlm_lexicon_decoder>`.
# In addition to the previously mentioned components, it also takes in various beam
# search decoding parameters and token/word parameters.
#

from torchaudio.prototype.ctc_decoder import kenlm_lexicon_decoder

LM_WEIGHT = 3.23
WORD_SCORE = -0.26

beam_search_decoder = kenlm_lexicon_decoder(
    lexicon=lexicon_file,
    tokens=tokens,
    kenlm=kenlm_file,
    nbest=3,
    beam_size=1500,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
)


######################################################################
# Greedy Decoder
# --------------
#
# For comparison against the beam search decoder, we also construct a
# basic greedy decoder.
#

from typing import List


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()


greedy_decoder = GreedyCTCDecoder(tokens)


######################################################################
# Run Inference
# -------------
#
# Now that we have the data, acoustic model, and decoder, we can perform
# inference. Recall the transcript corresponding to the waveform is
# ::
#   i really was very much afraid of showing him how much shocked i was at some parts of what he said
#

actual_transcript = "i really was very much afraid of showing him how much shocked i was at some parts of what he said"
actual_transcript = actual_transcript.split()

emission, _ = acoustic_model(waveform)


######################################################################
# The greedy decoder give the following result.
#

greedy_result = greedy_decoder(emission[0])
greedy_transcript = greedy_result
greedy_wer = torchaudio.functional.edit_distance(actual_transcript, greedy_transcript) / len(actual_transcript)

print(f"Transcript: {greedy_transcript}")
print(f"WER: {greedy_wer}")


######################################################################
# Using the beam search decoder:
#

beam_search_result = beam_search_decoder(emission)
beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
beam_search_wer = torchaudio.functional.edit_distance(actual_transcript, beam_search_result[0][0].words) / len(
    actual_transcript
)

print(f"Transcript: {beam_search_transcript}")
print(f"WER: {beam_search_wer}")


######################################################################
# We see that the transcript with the lexicon-constrained beam search
# decoder produces a more accurate result consisting of real words, while
# the greedy decoder can predict incorrectly spelled words like “affrayd”
# and “shoktd”.
#


######################################################################
# Beam Search Decoder Parameters
# ------------------------------
#
# In this section, we go a little bit more in depth about some different
# parameters and tradeoffs. For the full list of customizable parameters,
# please refer to the
# :py:func:`documentation <torchaudio.prototype.ctc_decoder.kenlm_lexicon_decoder>`.  # noqa
#


######################################################################
# Helper Function
# ~~~~~~~~~~~~~~~
#


def print_decoded(decoder, emission, param, param_value):
    start_time = time.monotonic()
    result = decoder(emission)
    decode_time = time.monotonic() - start_time

    transcript = " ".join(result[0][0].words).lower().strip()
    score = result[0][0].score
    print(f"{param} {param_value:<3}: {transcript} (score: {score:.2f}; {decode_time:.4f} secs)")


######################################################################
# nbest
# ~~~~~
#
# This parameter indicates the number of best Hypothesis to return, which
# is a property that is not possible with the greedy decoder. For
# instance, by setting ``nbest=3`` when constructing the beam search
# decoder earlier, we can now access the hypotheses with the top 3 scores.
#

for i in range(3):
    transcript = " ".join(beam_search_result[0][i].words).strip()
    score = beam_search_result[0][i].score
    print(f"{transcript} (score: {score})")


######################################################################
# beam size
# ~~~~~~~~~
#
# The ``beam_size`` parameter determines the maximum number of best
# hypotheses to hold after each decoding step. Using larger beam sizes
# allows for exploring a larger range of possible hypotheses which can
# produce hypotheses with higher scores, but it is computationally more
# expensive and does not provide additional gains beyond a certain point.
#
# In the example below, we see improvement in decoding quality as we
# increase beam size from 1 to 5 to 50, but notice how using a beam size
# of 500 provides the same output as beam size 50 while increase the
# computation time.
#

beam_sizes = [1, 5, 50, 500]

for beam_size in beam_sizes:
    beam_search_decoder = kenlm_lexicon_decoder(
        lexicon=lexicon_file,
        tokens=tokens,
        kenlm=kenlm_file,
        beam_size=beam_size,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam size", beam_size)


######################################################################
# beam size token
# ~~~~~~~~~~~~~~~
#
# The ``beam_size_token`` parameter corresponds to the number of tokens to
# consider for expanding each hypothesis at the decoding step. Exploring a
# larger number of next possible tokens increases the range of potential
# hypotheses at the cost of computation.
#

num_tokens = len(tokens)
beam_size_tokens = [1, 5, 10, num_tokens]

for beam_size_token in beam_size_tokens:
    beam_search_decoder = kenlm_lexicon_decoder(
        lexicon=lexicon_file,
        tokens=tokens,
        kenlm=kenlm_file,
        beam_size_token=beam_size_token,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam size token", beam_size_token)


######################################################################
# beam threshold
# ~~~~~~~~~~~~~~
#
# The ``beam_threshold`` parameter is used to prune the stored hypotheses
# set at each decoding step, removing hypotheses whose scores are greater
# than ``beam_threshold`` away from the highest scoring hypothesis. There
# is a balance between choosing smaller thresholds to prune more
# hypotheses and reduce the search space, and choosing a large enough
# threshold such that plausible hypotheses are not pruned.
#

beam_thresholds = [1, 5, 10, 25]

for beam_threshold in beam_thresholds:
    beam_search_decoder = kenlm_lexicon_decoder(
        lexicon=lexicon_file,
        tokens=tokens,
        kenlm=kenlm_file,
        beam_threshold=beam_threshold,
        lm_weight=LM_WEIGHT,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "beam threshold", beam_threshold)


######################################################################
# language model weight
# ~~~~~~~~~~~~~~~~~~~~~
#
# The ``lm_weight`` parameter is the weight to assign to the language
# model score which to accumulate with the acoustic model score for
# determining the overall scores. Larger weights encourage the model to
# predict next words based on the language model, while smaller weights
# give more weight to the acoustic model score instead.
#

lm_weights = [0, LM_WEIGHT, 15]

for lm_weight in lm_weights:
    beam_search_decoder = kenlm_lexicon_decoder(
        lexicon=lexicon_file,
        tokens=tokens,
        kenlm=kenlm_file,
        lm_weight=lm_weight,
        word_score=WORD_SCORE,
    )

    print_decoded(beam_search_decoder, emission, "lm weight", lm_weight)


######################################################################
# additional parameters
# ~~~~~~~~~~~~~~~~~~~~~
#
# Additional parameters that can be optimized include the following
#
# - ``word_score``: score to add when word finishes
# - ``unk_score``: unknown word appearance score to add
# - ``sil_score``: silence appearance score to add
# - ``log_add``: whether to use log add for lexicon Trie smearing
#
