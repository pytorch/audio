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

import os

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
# py:func:`torchaudio.pipelines`. For more detail on running Wav2Vec 2.0 speech
# recognition pipelines in torchaudio, please refer to `this
# tutorial <https://pytorch.org/audio/main/tutorials/speech_recognition_pipeline_tutorial.html>`__.
#

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
acoustic_model = bundle.get_model()


######################################################################
# We will load a sample from the LibriSpeech test-other dataset.
#

hub_dir = torch.hub.get_dir()

speech_url = "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/8461-258277-0000.wav"
speech_file = f"{hub_dir}/speech.wav"

torch.hub.download_url_to_file(speech_url, speech_file)

IPython.display.Audio(speech_file)


######################################################################
# The transcript corresponding to this audio file is
# ``"when it was the seven hundred and eighteenth night"``
#

waveform, sample_rate = torchaudio.load(speech_file)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)


######################################################################
# Files for Decoder
# ~~~~~~~~~~~~~~~~~
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
# including the blank and silent symbols.
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

token_url = "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/tokens-w2v2.txt"
token_file = f"{hub_dir}/token.txt"
torch.hub.download_url_to_file(token_url, token_file)


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

kenlm_url = "https://pytorch.s3.amazonaws.com/torchaudio/tutorial-assets/ctc-decoding/4-gram-librispeech.bin"
kenlm_file = f"{hub_dir}/kenlm.bin"
torch.hub.download_url_to_file(kenlm_url, kenlm_file)


######################################################################
# Construct Beam Search Decoder
# -----------------------------
#
# The decoder can be constructed using the
# :py:func:`torchaudio.prototype.ctc_decoder.kenlm_lexicon_decoder`
# factory function.
# In addition to the previously mentioned components, it also takes in
# various beam search decoding parameters and token/word parameters.
#

from torchaudio.prototype.ctc_decoder import kenlm_lexicon_decoder

beam_search_decoder = kenlm_lexicon_decoder(
    lexicon=lexicon_file,
    tokens=token_file,
    kenlm=kenlm_file,
    nbest=1,
    beam_size=1500,
    beam_size_token=50,
    lm_weight=3.23,
    word_score=-1.39,
    unk_score=float("-inf"),
    sil_score=0,
)


######################################################################
# Greedy Decoder
# --------------
#
# For comparison against the beam search decoder, we also construct a
# basic greedy decoder.
#


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


greedy_decoder = GreedyCTCDecoder(labels=bundle.get_labels())


######################################################################
# Run Inference
# -------------
#
# Now that we have the data, acoustic model, and decoder, we can perform
# inference. Recall the transcript corresponding to the waveform is
# ``"when it was the seven hundred and eighteenth night"``
#

emission, _ = acoustic_model(waveform)

######################################################################
# Using the beam search decoder:

beam_search_result = beam_search_decoder(emission)
beam_search_transcript = " ".join(beam_search_result[0][0].words).lower().strip()
print(beam_search_transcript)

######################################################################
# Using the greedy decoder:

greedy_result = greedy_decoder(emission[0])
greedy_transcript = greedy_result.replace("|", " ").lower().strip()
print(greedy_transcript)


######################################################################
# We see that the transcript with the lexicon-constrained beam search
# decoder consists of real words, while the greedy decoder can predict
# incorrectly spelled words like “hundrad”.
#
