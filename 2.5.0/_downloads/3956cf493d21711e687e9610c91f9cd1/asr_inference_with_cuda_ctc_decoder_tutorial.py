"""
ASR Inference with CUDA CTC Decoder
====================================

**Author**: `Yuekai Zhang <yuekaiz@nvidia.com>`__

This tutorial shows how to perform speech recognition inference using a
CUDA-based CTC beam search decoder.
We demonstrate this on a pretrained
`Zipformer <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_ctc>`__
model from `Next-gen Kaldi <https://nadirapovey.com/next-gen-kaldi-what-is-it>`__ project.

"""

######################################################################
# Overview
# --------
#
# Beam search decoding works by iteratively expanding text hypotheses (beams)
# with next possible characters, and maintaining only the hypotheses with the
# highest scores at each time step.
#
# The underlying implementation uses cuda to acclerate the whole decoding process
#  A mathematical formula for the decoder can be
# found in the `paper <https://arxiv.org/pdf/1408.2873.pdf>`__, and
# a more detailed algorithm can be found in this `blog
# <https://distill.pub/2017/ctc/>`__.
#
# Running ASR inference using a CUDA CTC Beam Search decoder
# requires the following components
#
# -  Acoustic Model: model predicting modeling units (BPE in this tutorial) from acoustic features
# -  BPE Model: the byte-pair encoding (BPE) tokenizer file
#


######################################################################
# Acoustic Model and Set Up
# -------------------------
#
# First we import the necessary utilities and fetch the data that we are
# working with
#

import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

######################################################################
#

import time
from pathlib import Path

import IPython
import sentencepiece as spm
from torchaudio.models.decoder import cuda_ctc_decoder
from torchaudio.utils import download_asset

######################################################################
#
# We use the pretrained
# `Zipformer <https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01>`__
# model that is trained on the `LibriSpeech
# dataset <http://www.openslr.org/12>`__. The model is jointly trained with CTC and Transducer loss functions.
# In this tutorial, we only use CTC head of the model.


def download_asset_external(url, key):
    path = Path(torch.hub.get_dir()) / "torchaudio" / Path(key)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, path)
    return str(path)


url_prefix = "https://huggingface.co/Zengwei/icefall-asr-librispeech-pruned-transducer-stateless7-ctc-2022-12-01"
model_link = f"{url_prefix}/resolve/main/exp/cpu_jit.pt"
model_path = download_asset_external(model_link, "cuda_ctc_decoder/cpu_jit.pt")


######################################################################
# We will load a sample from the LibriSpeech test-other dataset.
#

speech_file = download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")
waveform, sample_rate = torchaudio.load(speech_file)
assert sample_rate == 16000
IPython.display.Audio(speech_file)


######################################################################
# The transcript corresponding to this audio file is
#
# .. code-block::
#
#    i really was very much afraid of showing him how much shocked i was at some parts of what he said
#


######################################################################
# Files and Data for Decoder
# --------------------------
#
# Next, we load in our token from BPE model, which is the tokenizer for decoding.
#


######################################################################
# Tokens
# ~~~~~~
#
# The tokens are the possible symbols that the acoustic model can predict,
# including the blank symbol in CTC. In this tutorial, it includes 500 BPE tokens.
# It can either be passed in as a
# file, where each line consists of the tokens corresponding to the same
# index, or as a list of tokens, each mapping to a unique index.
#
# .. code-block::
#
#    # tokens
#    <blk>
#    <sos/eos>
#    <unk>
#    S
#    _THE
#    _A
#    T
#    _AND
#    ...
#
bpe_link = f"{url_prefix}/resolve/main/data/lang_bpe_500/bpe.model"
bpe_path = download_asset_external(bpe_link, "cuda_ctc_decoder/bpe.model")

bpe_model = spm.SentencePieceProcessor()
bpe_model.load(bpe_path)
tokens = [bpe_model.id_to_piece(id) for id in range(bpe_model.get_piece_size())]
print(tokens)

######################################################################
# Construct CUDA Decoder
# ----------------------
# In this tutorial, we will construct a CUDA beam search decoder.
# The decoder can be constructed using the factory function
# :py:func:`~torchaudio.models.decoder.cuda_ctc_decoder`.
#

cuda_decoder = cuda_ctc_decoder(tokens, nbest=10, beam_size=10, blank_skip_threshold=0.95)
######################################################################
# Run Inference
# -------------
#
# Now that we have the data, acoustic model, and decoder, we can perform
# inference. The output of the beam search decoder is of type
# :py:class:`~torchaudio.models.decoder.CUCTCHypothesis`, consisting of the
# predicted token IDs, words (symbols corresponding to the token IDs), and hypothesis scores.
# Recall the transcript corresponding to the
# waveform is
#
# .. code-block::
#
#    i really was very much afraid of showing him how much shocked i was at some parts of what he said
#

actual_transcript = "i really was very much afraid of showing him how much shocked i was at some parts of what he said"
actual_transcript = actual_transcript.split()

device = torch.device("cuda", 0)
acoustic_model = torch.jit.load(model_path)
acoustic_model.to(device)
acoustic_model.eval()

waveform = waveform.to(device)

feat = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, snip_edges=False)
feat = feat.unsqueeze(0)
feat_lens = torch.tensor(feat.size(1), device=device).unsqueeze(0)

encoder_out, encoder_out_lens = acoustic_model.encoder(feat, feat_lens)
nnet_output = acoustic_model.ctc_output(encoder_out)
log_prob = torch.nn.functional.log_softmax(nnet_output, -1)

print(f"The shape of log_prob: {log_prob.shape}, the shape of encoder_out_lens: {encoder_out_lens.shape}")

######################################################################
# The cuda ctc decoder gives the following result.
#

results = cuda_decoder(log_prob, encoder_out_lens.to(torch.int32))
beam_search_transcript = bpe_model.decode(results[0][0].tokens).lower()
beam_search_wer = torchaudio.functional.edit_distance(actual_transcript, beam_search_transcript.split()) / len(
    actual_transcript
)

print(f"Transcript: {beam_search_transcript}")
print(f"WER: {beam_search_wer}")

######################################################################
# Beam Search Decoder Parameters
# ------------------------------
#
# In this section, we go a little bit more in depth about some different
# parameters and tradeoffs. For the full list of customizable parameters,
# please refer to the
# :py:func:`documentation <torchaudio.models.decoder.cuda_ctc_decoder>`.
#


######################################################################
# Helper Function
# ~~~~~~~~~~~~~~~
#


def print_decoded(cuda_decoder, bpe_model, log_prob, encoder_out_lens, param, param_value):
    start_time = time.monotonic()
    results = cuda_decoder(log_prob, encoder_out_lens.to(torch.int32))
    decode_time = time.monotonic() - start_time
    transcript = bpe_model.decode(results[0][0].tokens).lower()
    score = results[0][0].score
    print(f"{param} {param_value:<3}: {transcript} (score: {score:.2f}; {decode_time:.4f} secs)")


######################################################################
# nbest
# ~~~~~
#
# This parameter indicates the number of best hypotheses to return. For
# instance, by setting ``nbest=10`` when constructing the beam search
# decoder earlier, we can now access the hypotheses with the top 10 scores.
#

for i in range(10):
    transcript = bpe_model.decode(results[0][i].tokens).lower()
    score = results[0][i].score
    print(f"{transcript} (score: {score})")


######################################################################
# beam size
# ~~~~~~~~~
#
# The ``beam_size`` parameter determines the maximum number of best
# hypotheses to hold after each decoding step. Using larger beam sizes
# allows for exploring a larger range of possible hypotheses which can
# produce hypotheses with higher scores, but it does not provide additional gains beyond a certain point.
# We recommend to set beam_size=10 for cuda beam search decoder.
#
# In the example below, we see improvement in decoding quality as we
# increase beam size from 1 to 3, but notice how using a beam size
# of 3 provides the same output as beam size 10.
#

beam_sizes = [1, 2, 3, 10]

for beam_size in beam_sizes:
    beam_search_decoder = cuda_ctc_decoder(
        tokens,
        nbest=1,
        beam_size=beam_size,
        blank_skip_threshold=0.95,
    )
    print_decoded(beam_search_decoder, bpe_model, log_prob, encoder_out_lens, "beam size", beam_size)


######################################################################
# blank skip threshold
# ~~~~~~~~~~~~~~~~~~~~
#
# The ``blank_skip_threshold`` parameter is used to prune the frames which have large blank probability.
# Pruning these frames with a good blank_skip_threshold could speed up decoding
# process a lot while no accuracy drop.
# Since the rule of CTC, we would keep at least one blank frame between two non-blank frames
# to avoid mistakenly merge two consecutive identical symbols.
# We recommend to set blank_skip_threshold=0.95 for cuda beam search decoder.
#

blank_skip_probs = [0.25, 0.95, 1.0]

for blank_skip_prob in blank_skip_probs:
    beam_search_decoder = cuda_ctc_decoder(
        tokens,
        nbest=10,
        beam_size=10,
        blank_skip_threshold=blank_skip_prob,
    )
    print_decoded(beam_search_decoder, bpe_model, log_prob, encoder_out_lens, "blank_skip_threshold", blank_skip_prob)

del cuda_decoder

######################################################################
# Benchmark with flashlight CPU decoder
# -------------------------------------
# We benchmark the throughput and accuracy between CUDA decoder and CPU decoder using librispeech test_other set.
# To reproduce below benchmark results, you may refer `here <https://github.com/pytorch/audio/tree/main/examples/asr/librispeech_cuda_ctc_decoder>`__.
#
# +--------------+------------------------------------------+---------+-----------------------+-----------------------------+
# | Decoder      | Setting                                  | WER (%) | N-Best Oracle WER (%) | Decoder Cost Time (seconds) |
# +==============+==========================================+=========+=======================+=============================+
# | CUDA decoder | blank_skip_threshold 0.95                | 5.81    | 4.11                  | 2.57                        |
# +--------------+------------------------------------------+---------+-----------------------+-----------------------------+
# | CUDA decoder | blank_skip_threshold 1.0 (no frame-skip) | 5.81    | 4.09                  | 6.24                        |
# +--------------+------------------------------------------+---------+-----------------------+-----------------------------+
# | CPU decoder  | beam_size_token 10                       | 5.86    | 4.30                  | 28.61                       |
# +--------------+------------------------------------------+---------+-----------------------+-----------------------------+
# | CPU decoder  | beam_size_token 500                      | 5.86    | 4.30                  | 791.80                      |
# +--------------+------------------------------------------+---------+-----------------------+-----------------------------+
#
# From the above table, CUDA decoder could give a slight improvement in WER and a significant increase in throughput.
