#include <torchaudio/csrc/rnnt/workspace.h>

namespace torchaudio {
namespace rnnt {

void IntWorkspace::ResetAlphaBetaCounters() {
  if (data_ != nullptr && options_.device_ == GPU) {
    cudaMemset(
        GetPointerToAlphaCounters(),
        0,
        ComputeSizeForAlphaCounters(options_) * sizeof(int));
    cudaMemset(
        GetPointerToBetaCounters(),
        0,
        ComputeSizeForBetaCounters(options_) * sizeof(int));
  }
}

} // namespace rnnt
} // namespace torchaudio
