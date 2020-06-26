#include <sox.h>
#include <torchaudio/csrc/sox_effects.h>

using namespace torch::indexing;

namespace torchaudio {
namespace sox_effects {

namespace {

enum SoxEffectsResourceState { NotInitialized, Initialized, ShutDown };
SoxEffectsResourceState SOX_RESOURCE_STATE = NotInitialized;

} // namespace

void initialize_sox_effects() {
  if (SOX_RESOURCE_STATE == ShutDown) {
    throw std::runtime_error(
        "SoX Effects has been shut down. Cannot initialize again.");
  }
  if (SOX_RESOURCE_STATE == NotInitialized) {
    if (sox_init() != SOX_SUCCESS) {
      throw std::runtime_error("Failed to initialize sox effects.");
    };
    SOX_RESOURCE_STATE = Initialized;
  }
};

void shutdown_sox_effects() {
  if (SOX_RESOURCE_STATE == NotInitialized) {
    throw std::runtime_error(
        "SoX Effects is not initialized. Cannot shutdown.");
  }
  if (SOX_RESOURCE_STATE == Initialized) {
    if (sox_quit() != SOX_SUCCESS) {
      throw std::runtime_error("Failed to initialize sox effects.");
    };
    SOX_RESOURCE_STATE = ShutDown;
  }
}

std::vector<std::string> list_effects() {
  std::vector<std::string> names;
  const sox_effect_fn_t* fns = sox_get_effect_fns();
  for (int i = 0; fns[i]; ++i) {
    const sox_effect_handler_t* handler = fns[i]();
    if (handler && handler->name)
      names.push_back(handler->name);
  }
  return names;
}

} // namespace sox_effects
} // namespace torchaudio
