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

static auto registerSetSoxOptions =
    torch::RegisterOperators()
        .op("torchaudio::sox_utils_set_seed", &sox_utils::set_seed)
        .op("torchaudio::sox_utils_set_verbosity", &sox_utils::set_verbosity)
        .op("torchaudio::sox_utils_set_use_threads",
            &sox_utils::set_use_threads)
        .op("torchaudio::sox_utils_set_buffer_size",
            &sox_utils::set_buffer_size)
        .op("torchaudio::sox_utils_list_effects", &sox_utils::list_effects)
        .op("torchaudio::sox_utils_list_read_formats",
            &sox_utils::list_read_formats)
        .op("torchaudio::sox_utils_list_write_formats",
            &sox_utils::list_write_formats);

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
            "torchaudio::sox_io_save_audio_file(str path, __torch__.torch.classes.torchaudio.TensorSignal signal, float compression) -> ()")
        .catchAllKernel<
            decltype(sox_io::save_audio_file),
            &sox_io::save_audio_file>());

////////////////////////////////////////////////////////////////////////////////
// sox_effects.h
////////////////////////////////////////////////////////////////////////////////
static auto registerSoxEffects =
    torch::RegisterOperators()
        .op("torchaudio::sox_effects_initialize_sox_effects",
            &sox_effects::initialize_sox_effects)
        .op("torchaudio::sox_effects_shutdown_sox_effects",
            &sox_effects::shutdown_sox_effects)
        .op(torch::RegisterOperators::options()
                .schema(
                    "torchaudio::sox_effects_apply_effects_tensor(__torch__.torch.classes.torchaudio.TensorSignal input_signal, str[][] effects) -> __torch__.torch.classes.torchaudio.TensorSignal output_signal")
                .catchAllKernel<
                    decltype(sox_effects::apply_effects_tensor),
                    &sox_effects::apply_effects_tensor>())
        .op(torch::RegisterOperators::options()
                .schema(
                    "torchaudio::sox_effects_apply_effects_file(str path, str[][] effects, bool normalize, bool channels_first) -> __torch__.torch.classes.torchaudio.TensorSignal output_signal")
                .catchAllKernel<
                    decltype(sox_effects::apply_effects_file),
                    &sox_effects::apply_effects_file>());

} // namespace
} // namespace torchaudio
#endif
