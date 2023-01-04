#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader_wrapper.h>
#include <stdexcept>

namespace torchaudio {
namespace ffmpeg {

namespace {

c10::intrusive_ptr<StreamReaderBinding> init(
    const std::string& src,
    const c10::optional<std::string>& device,
    const c10::optional<OptionDict>& option) {
  return c10::make_intrusive<StreamReaderBinding>(
      get_input_format_context(src, device, option));
}

using S = const c10::intrusive_ptr<StreamReaderBinding>&;

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::ffmpeg_init", []() { avdevice_register_all(); });
  m.def("torchaudio::ffmpeg_get_log_level", []() -> int64_t {
    return static_cast<int64_t>(av_log_get_level());
  });
  m.def("torchaudio::ffmpeg_set_log_level", [](int64_t level) {
    av_log_set_level(static_cast<int>(level));
  });
  m.class_<StreamReaderBinding>("ffmpeg_StreamReader")
      .def(torch::init<>(init))
      .def("num_src_streams", [](S self) { return self->num_src_streams(); })
      .def("num_out_streams", [](S self) { return self->num_out_streams(); })
      .def("get_metadata", [](S self) { return self->get_metadata(); })
      .def(
          "get_src_stream_info",
          [](S s, int64_t i) { return s->get_src_stream_info(i); })
      .def(
          "get_out_stream_info",
          [](S s, int64_t i) { return s->get_out_stream_info(i); })
      .def(
          "find_best_audio_stream",
          [](S s) { return s->find_best_audio_stream(); })
      .def(
          "find_best_video_stream",
          [](S s) { return s->find_best_video_stream(); })
      .def("seek", [](S s, double t, int64_t mode) { return s->seek(t, mode); })
      .def(
          "add_audio_stream",
          [](S s,
             int64_t i,
             int64_t frames_per_chunk,
             int64_t num_chunks,
             const c10::optional<std::string>& filter_desc,
             const c10::optional<std::string>& decoder,
             const c10::optional<OptionDict>& decoder_option) {
            s->add_audio_stream(
                i,
                frames_per_chunk,
                num_chunks,
                filter_desc,
                decoder,
                decoder_option);
          })
      .def(
          "add_video_stream",
          [](S s,
             int64_t i,
             int64_t frames_per_chunk,
             int64_t num_chunks,
             const c10::optional<std::string>& filter_desc,
             const c10::optional<std::string>& decoder,
             const c10::optional<OptionDict>& decoder_option,
             const c10::optional<std::string>& hw_accel) {
            s->add_video_stream(
                i,
                frames_per_chunk,
                num_chunks,
                filter_desc,
                decoder,
                decoder_option,
                hw_accel);
          })
      .def("remove_stream", [](S s, int64_t i) { s->remove_stream(i); })
      .def(
          "process_packet",
          [](S s, const c10::optional<double>& timeout, const double backoff) {
            return s->process_packet(timeout, backoff);
          })
      .def("process_all_packets", [](S s) { s->process_all_packets(); })
      .def(
          "fill_buffer",
          [](S s, const c10::optional<double>& timeout, const double backoff) {
            return s->fill_buffer(timeout, backoff);
          })
      .def("is_buffer_ready", [](S s) { return s->is_buffer_ready(); })
      .def("pop_chunks", [](S s) { return s->pop_chunks(); });
}

} // namespace
} // namespace ffmpeg
} // namespace torchaudio
