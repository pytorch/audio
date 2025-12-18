#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(_torchaudio, m) {
  m.def("is_align_available() -> bool");
  m.def("cuda_version() -> int?");
}

STABLE_TORCH_LIBRARY_IMPL(_torchaudio, CompositeExplicitAutograd, m) {
  m.impl("is_align_available", TORCH_BOX(&torchaudio::is_align_available));
  m.impl("cuda_version", TORCH_BOX(&torchaudio::cuda_version));
}
