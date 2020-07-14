#include <sox.h>
#include <torchaudio/csrc/sox_effects.h>

namespace torchaudio {
namespace sox_effects {

namespace {

enum SoxEffectsResourceState { NotInitialized, Initialized, ShutDown };
SoxEffectsResourceState SOX_RESOURCE_STATE = NotInitialized;
std::mutex SOX_RESOUCE_STATE_MUTEX;

} // namespace

void initialize_sox_effects() {
  const std::lock_guard<std::mutex> lock(SOX_RESOUCE_STATE_MUTEX);

  switch (SOX_RESOURCE_STATE) {
    case NotInitialized:
      if (sox_init() != SOX_SUCCESS) {
        throw std::runtime_error("Failed to initialize sox effects.");
      };
      SOX_RESOURCE_STATE = Initialized;
    case Initialized:
      break;
    case ShutDown:
      throw std::runtime_error(
          "SoX Effects has been shut down. Cannot initialize again.");
  }
};

void shutdown_sox_effects() {
  const std::lock_guard<std::mutex> lock(SOX_RESOUCE_STATE_MUTEX);

  switch (SOX_RESOURCE_STATE) {
    case NotInitialized:
      throw std::runtime_error(
          "SoX Effects is not initialized. Cannot shutdown.");
    case Initialized:
      if (sox_quit() != SOX_SUCCESS) {
        throw std::runtime_error("Failed to initialize sox effects.");
      };
      SOX_RESOURCE_STATE = ShutDown;
    case ShutDown:
      break;
  }
}

} // namespace sox_effects
} // namespace torchaudio
