#include <torch/extension.h>
#include <torchaudio/csrc/sox/utils.h>

namespace torchaudio {
namespace sox {
namespace {

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
