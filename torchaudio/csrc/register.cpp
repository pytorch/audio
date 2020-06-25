#ifndef TORCHAUDIO_REGISTER_H
#define TORCHAUDIO_REGISTER_H

#include <torchaudio/csrc/sox_effects.h>
#include <torchaudio/csrc/sox_io.h>
#include <torchaudio/csrc/typedefs.h>

namespace torchaudio {
namespace {

static auto registerSignalInfo =
    torch::class_<SignalInfo>("torchaudio", "SignalInfo")
        .def(torch::init<int64_t, int64_t, int64_t>())
        .def("get_sample_rate", &SignalInfo::getSampleRate)
        .def("get_num_channels", &SignalInfo::getNumChannels)
        .def("get_num_frames", &SignalInfo::getNumFrames);

static auto registerGetInfo = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "torchaudio::sox_io_get_info(str path) -> __torch__.torch.classes.torchaudio.SignalInfo info")
        .catchAllKernel<decltype(sox_io::get_info), &sox_io::get_info>());

////////////////////////////////////////////////////////////////////////////////
// sox_effects.h
////////////////////////////////////////////////////////////////////////////////
static auto registerSoxEffects =
    torch::RegisterOperators(
        "torchaudio::sox_effects_initialize_sox_effects",
        &sox_effects::initialize_sox_effects)
        .op("torchaudio::sox_effects_shutdown_sox_effects",
            &sox_effects::shutdown_sox_effects)
        .op("torchaudio::sox_effects_list_effects", &sox_effects::list_effects);

} // namespace
} // namespace torchaudio
#endif
