#ifndef TORCHAUDIO_SOX_EFFECT_H
#define TORCHAUDIO_SOX_EFFECT_H

#include <sox.h>
#include <torch/extension.h>

namespace torch {
namespace audio {
// get names of all sox effects
std::vector<std::string> get_effect_names();

// Initialize and Shutdown SoX effects chain.  These functions should only be
// run once.
int initialize_sox();
int shutdown_sox();

// Struct for build_flow_effects function
struct SoxEffect {
  SoxEffect() : ename(""), eopts({""}) {}
  std::string ename;
  std::vector<std::string> eopts;
};

/// Build a SoX chain, flow the effects, and capture the results in a tensor.
/// An audio file from the given `path` flows through an effects chain given
/// by a list of effects and effect options to an output buffer which is encoded
/// into memory to a target signal type and target signal encoding.  The
/// resulting buffer is then placed into a tensor.  This function returns the
/// output tensor and the sample rate of the output tensor.
int build_flow_effects(
    const std::string& file_name,
    at::Tensor otensor,
    bool ch_first,
    sox_signalinfo_t* target_signal,
    sox_encodinginfo_t* target_encoding,
    const char* file_type,
    std::vector<SoxEffect> pyeffs,
    int max_num_eopts);
} // namespace audio
} // namespace torch

#endif
