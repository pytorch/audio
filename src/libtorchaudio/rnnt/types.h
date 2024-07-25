#pragma once

namespace torchaudio {
namespace rnnt {

enum status_t {
  SUCCESS = 0,
  FAILURE = 1,
  COMPUTE_DENOMINATOR_REDUCE_MAX_FAILED = 2,
  COMPUTE_DENOMINATOR_REDUCE_SUM_FAILED = 3,
  COMPUTE_LOG_PROBS_FAILED = 4,
  COMPUTE_ALPHAS_BETAS_COSTS_FAILED = 5,
  COMPUTE_GRADIENTS_FAILED = 6
};

enum device_t { UNDEFINED = 0, CPU = 1, GPU = 2 };

} // namespace rnnt
} // namespace torchaudio
