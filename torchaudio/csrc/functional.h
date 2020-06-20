#ifndef TORCHAUDIO_FUNCTIONAL_H
#define TORCHAUDIO_FUNCTINOAL_H

#include <torch/script.h>

namespace torchaudio {
namespace functional {

torch::Tensor lfilter(
    torch::Tensor waveform,
    torch::Tensor a_coeffs,
    torch::Tensor b_coeffs,
    const bool clamp);

} // namespace functional
} // namespace torchaudio

#endif
