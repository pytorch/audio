#ifndef TORCHAUDIO_REGISTER_H
#define TORCHAUDIO_REGISTER_H

#include <torchaudio/csrc/typedefs.h>

namespace torchaudio {
namespace {

static auto registerSignalInfo =
    torch::class_<SignalInfo>("torchaudio", "SignalInfo")
        .def(torch::init<int64_t, int64_t, int64_t>())
        .def("get_sample_rate", &SignalInfo::getSampleRate)
        .def("get_num_channels", &SignalInfo::getNumChannels)
        .def("get_num_samples", &SignalInfo::getNumSamples);

} // namespace
} // namespace torchaudio
#endif
