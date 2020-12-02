#include <torchaudio/csrc/kaldi/kaldi.h>

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  //////////////////////////////////////////////////////////////////////////////
  // kaldi.h
  //////////////////////////////////////////////////////////////////////////////
  m.def(
      "torchaudio::kaldi_ComputeKaldiPitch",
      &torchaudio::kaldi::ComputeKaldiPitch);
}
