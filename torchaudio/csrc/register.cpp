#ifndef TORCHAUDIO_REGISTER_H
#define TORCHAUDIO_REGISTER_H

#include <torchaudio/csrc/sox_effects.h>
#include <torchaudio/csrc/sox_io.h>
#include <torchaudio/csrc/sox_utils.h>

namespace torchaudio {
namespace {

////////////////////////////////////////////////////////////////////////////////
// sox_utils.h
////////////////////////////////////////////////////////////////////////////////
static auto registerTensorSignal =
    torch::class_<sox_utils::TensorSignal>("torchaudio", "TensorSignal")
        .def(torch::init<torch::Tensor, int64_t, bool>())
        .def("get_tensor", &sox_utils::TensorSignal::getTensor)
        .def("get_sample_rate", &sox_utils::TensorSignal::getSampleRate)
        .def("get_channels_first", &sox_utils::TensorSignal::getChannelsFirst);

////////////////////////////////////////////////////////////////////////////////
// sox_io.h
////////////////////////////////////////////////////////////////////////////////
static auto registerSignalInfo =
    torch::class_<sox_io::SignalInfo>("torchaudio", "SignalInfo")
        .def("get_sample_rate", &sox_io::SignalInfo::getSampleRate)
        .def("get_num_channels", &sox_io::SignalInfo::getNumChannels)
        .def("get_num_frames", &sox_io::SignalInfo::getNumFrames);

static auto registerGetInfo = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "torchaudio::sox_io_get_info(str path) -> __torch__.torch.classes.torchaudio.SignalInfo info")
        .catchAllKernel<decltype(sox_io::get_info), &sox_io::get_info>());

static auto registerLoadAudioFile = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "torchaudio::sox_io_load_audio_file(str path, int frame_offset, int num_frames, bool normalize, bool channels_first) -> __torch__.torch.classes.torchaudio.TensorSignal signal")
        .catchAllKernel<
            decltype(sox_io::load_audio_file),
            &sox_io::load_audio_file>());

static auto registerSaveAudioFile = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "torchaudio::sox_io_save_audio_file(str path, __torch__.torch.classes.torchaudio.TensorSignal signal, float compression, int frames_per_chunk) -> ()")
        .catchAllKernel<
            decltype(sox_io::save_audio_file),
            &sox_io::save_audio_file>());

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
