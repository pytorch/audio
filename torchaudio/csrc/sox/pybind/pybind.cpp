#include <torch/extension.h>

#include <torchaudio/csrc/sox/pybind/effects.h>
#include <torchaudio/csrc/sox/pybind/io.h>

PYBIND11_MODULE(_torchaudio_sox, m) {
  m.def(
      "get_info_fileobj",
      &torchaudio::sox_io::get_info_fileobj,
      "Get metadata of audio in file object.");
  m.def(
      "load_audio_fileobj",
      &torchaudio::sox_io::load_audio_fileobj,
      "Load audio from file object.");
  m.def(
      "save_audio_fileobj",
      &torchaudio::sox_io::save_audio_fileobj,
      "Save audio to file obj.");
  m.def(
      "apply_effects_fileobj",
      &torchaudio::sox_effects::apply_effects_fileobj,
      "Decode audio data from file-like obj and apply effects.");
}
