#include <libtorchaudio/sox/effects.h>
#include <libtorchaudio/sox/io.h>
#include <libtorchaudio/sox/utils.h>
#include <torch/extension.h>

namespace torchaudio {
namespace sox {
namespace {

TORCH_LIBRARY(torchaudio_sox, m) {
  m.def("torchaudio_sox::get_info", &get_info_file);
  m.def("torchaudio_sox::load_audio_file", &load_audio_file);
  m.def("torchaudio_sox::save_audio_file", &save_audio_file);
  m.def("torchaudio_sox::initialize_sox_effects", &initialize_sox_effects);
  m.def("torchaudio_sox::shutdown_sox_effects", &shutdown_sox_effects);
  m.def("torchaudio_sox::apply_effects_tensor", &apply_effects_tensor);
  m.def("torchaudio_sox::apply_effects_file", &apply_effects_file);
}

PYBIND11_MODULE(_torchaudio_sox, m) {
  m.def("set_seed", &set_seed, "Set random seed.");
  m.def("set_verbosity", &set_verbosity, "Set verbosity.");
  m.def("set_use_threads", &set_use_threads, "Set threading.");
  m.def("set_buffer_size", &set_buffer_size, "Set buffer size.");
  m.def("get_buffer_size", &get_buffer_size, "Get buffer size.");
  m.def("list_effects", &list_effects, "List available effects.");
  m.def(
      "list_read_formats",
      &list_read_formats,
      "List supported formats for decoding.");
  m.def(
      "list_write_formats",
      &list_write_formats,
      "List supported formats for encoding.");
}

} // namespace
} // namespace sox
} // namespace torchaudio
