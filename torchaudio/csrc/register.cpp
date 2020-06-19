#ifndef TORCHAUDIO_REGISTER_H
#define TORCHAUDIO_REGISTER_H

#include <torchaudio/csrc/sox_io.h>
#include <torchaudio/csrc/typedefs.h>

namespace torchaudio {
namespace {

static auto registerSignalInfo =
    torch::class_<SignalInfo>("torchaudio", "SignalInfo")
        .def(torch::init<int64_t, int64_t, int64_t>())
        .def("get_sample_rate", &SignalInfo::getSampleRate)
        .def("get_num_channels", &SignalInfo::getNumChannels)
        .def("get_num_samples", &SignalInfo::getNumSamples);

static auto registerGetInfo = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "torchaudio::sox_io_get_info(str path) -> __torch__.torch.classes.torchaudio.SignalInfo info")
        .catchAllKernel<decltype(sox_io::get_info), &sox_io::get_info>());

} // namespace
} // namespace torchaudio
#endif
