#ifndef TORCHAUDIO_SOX_EFFECTS_H
#define TORCHAUDIO_SOX_EFFECTS_H

#include <torch/script.h>

namespace torchaudio {
namespace sox_effects {

void initialize_sox_effects();

void shutdown_sox_effects();

std::vector<std::string> list_effects();

} // namespace sox_effects
} // namespace torchaudio

#endif
