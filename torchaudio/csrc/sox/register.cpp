#include <torchaudio/csrc/sox/effects.h>
#include <torchaudio/csrc/sox/io.h>
#include <torchaudio/csrc/sox/utils.h>

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  //////////////////////////////////////////////////////////////////////////////
  // sox_utils.h
  //////////////////////////////////////////////////////////////////////////////
  m.class_<torchaudio::sox_utils::TensorSignal>("TensorSignal")
      .def(torch::init<torch::Tensor, int64_t, bool>())
      .def("get_tensor", &torchaudio::sox_utils::TensorSignal::getTensor)
      .def(
          "get_sample_rate",
          &torchaudio::sox_utils::TensorSignal::getSampleRate)
      .def(
          "get_channels_first",
          &torchaudio::sox_utils::TensorSignal::getChannelsFirst);

  m.def("torchaudio::sox_utils_set_seed", &torchaudio::sox_utils::set_seed);
  m.def(
      "torchaudio::sox_utils_set_verbosity",
      &torchaudio::sox_utils::set_verbosity);
  m.def(
      "torchaudio::sox_utils_set_use_threads",
      &torchaudio::sox_utils::set_use_threads);
  m.def(
      "torchaudio::sox_utils_set_buffer_size",
      &torchaudio::sox_utils::set_buffer_size);
  m.def(
      "torchaudio::sox_utils_list_effects",
      &torchaudio::sox_utils::list_effects);
  m.def(
      "torchaudio::sox_utils_list_read_formats",
      &torchaudio::sox_utils::list_read_formats);
  m.def(
      "torchaudio::sox_utils_list_write_formats",
      &torchaudio::sox_utils::list_write_formats);

  //////////////////////////////////////////////////////////////////////////////
  // sox_io.h
  //////////////////////////////////////////////////////////////////////////////
  m.class_<torchaudio::sox_io::SignalInfo>("SignalInfo")
      .def("get_sample_rate", &torchaudio::sox_io::SignalInfo::getSampleRate)
      .def("get_num_channels", &torchaudio::sox_io::SignalInfo::getNumChannels)
      .def("get_num_frames", &torchaudio::sox_io::SignalInfo::getNumFrames)
      .def(
          "get_bits_per_sample",
          &torchaudio::sox_io::SignalInfo::getBitsPerSample);

  m.def("torchaudio::sox_io_get_info", &torchaudio::sox_io::get_info_file);
  m.def(
      "torchaudio::sox_io_load_audio_file("
      "str path,"
      "int? frame_offset=None,"
      "int? num_frames=None,"
      "bool? normalize=True,"
      "bool? channels_first=False,"
      "str? format=None"
      ") -> __torch__.torch.classes.torchaudio.TensorSignal",
      &torchaudio::sox_io::load_audio_file);
  m.def(
      "torchaudio::sox_io_save_audio_file",
      &torchaudio::sox_io::save_audio_file);

  //////////////////////////////////////////////////////////////////////////////
  // sox_effects.h
  //////////////////////////////////////////////////////////////////////////////
  m.def(
      "torchaudio::sox_effects_initialize_sox_effects",
      &torchaudio::sox_effects::initialize_sox_effects);
  m.def(
      "torchaudio::sox_effects_shutdown_sox_effects",
      &torchaudio::sox_effects::shutdown_sox_effects);
  m.def(
      "torchaudio::sox_effects_apply_effects_tensor",
      &torchaudio::sox_effects::apply_effects_tensor);
  m.def(
      "torchaudio::sox_effects_apply_effects_file",
      &torchaudio::sox_effects::apply_effects_file);
}
