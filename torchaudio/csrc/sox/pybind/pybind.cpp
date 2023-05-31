#include <torch/extension.h>

#include <torchaudio/csrc/sox/pybind/effects.h>
#include <torchaudio/csrc/sox/pybind/io.h>

namespace torchaudio::sox {
namespace {
PYBIND11_MODULE(_torchaudio_sox, m) {
  m.def(
      "get_info_fileobj",
      &get_info_fileobj,
      "Get metadata of audio in file object.");
  m.def(
      "load_audio_fileobj",
      &load_audio_fileobj,
      "Load audio from file object.");
  m.def("save_audio_fileobj", &save_audio_fileobj, "Save audio to file obj.");
  m.def(
      "apply_effects_fileobj",
      &apply_effects_fileobj,
      "Decode audio data from file-like obj and apply effects.");
}
} // namespace
} // namespace torchaudio::sox
