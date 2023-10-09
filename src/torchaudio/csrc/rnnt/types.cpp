#include <libtorchaudio/rnnt/types.h>

namespace torchaudio {
namespace rnnt {

const char* toString(status_t status) {
  switch (status) {
    case SUCCESS:
      return "success";
    case FAILURE:
      return "failure";
    case COMPUTE_DENOMINATOR_REDUCE_MAX_FAILED:
      return "compute_denominator_reduce_max_failed";
    case COMPUTE_DENOMINATOR_REDUCE_SUM_FAILED:
      return "compute_denominator_reduce_sum_failed";
    case COMPUTE_LOG_PROBS_FAILED:
      return "compute_log_probs_failed";
    case COMPUTE_ALPHAS_BETAS_COSTS_FAILED:
      return "compute_alphas_betas_costs_failed";
    case COMPUTE_GRADIENTS_FAILED:
      return "compute_gradients_failed";
    default:
      return "unknown";
  }
}

const char* toString(device_t device) {
  switch (device) {
    case UNDEFINED:
      return "undefined";
    case CPU:
      return "cpu";
    case GPU:
      return "gpu";
    default:
      return "unknown";
  }
}

} // namespace rnnt
} // namespace torchaudio
