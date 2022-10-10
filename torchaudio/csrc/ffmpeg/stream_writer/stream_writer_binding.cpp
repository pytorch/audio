#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer_wrapper.h>

namespace torchaudio {
namespace ffmpeg {
namespace {

c10::intrusive_ptr<StreamWriterBinding> init(
    const std::string& dst,
    const c10::optional<std::string>& format) {
  return c10::make_intrusive<StreamWriterBinding>(
      get_output_format_context(dst, format));
}

using S = const c10::intrusive_ptr<StreamWriterBinding>&;

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.class_<StreamWriterBinding>("ffmpeg_StreamWriter")
      .def(torch::init<>(init))
      .def(
          "add_audio_stream",
          [](S s,
             int64_t sample_rate,
             int64_t num_channels,
             const std::string& format,
             const c10::optional<std::string>& encoder,
             const c10::optional<OptionDict>& encoder_option,
             const c10::optional<std::string>& encoder_format) {
            s->add_audio_stream(
                sample_rate,
                num_channels,
                format,
                encoder,
                encoder_option,
                encoder_format);
          })
      .def(
          "add_video_stream",
          [](S s,
             double frame_rate,
             int64_t width,
             int64_t height,
             const std::string& format,
             const c10::optional<std::string>& encoder,
             const c10::optional<OptionDict>& encoder_option,
             const c10::optional<std::string>& encoder_format,
             const c10::optional<std::string>& hw_accel) {
            s->add_video_stream(
                frame_rate,
                width,
                height,
                format,
                encoder,
                encoder_option,
                encoder_format,
                hw_accel);
          })
      .def(
          "set_metadata",
          [](S s, const OptionDict& metadata) { s->set_metadata(metadata); })
      .def("dump_format", [](S s, int64_t i) { s->dump_format(i); })
      .def(
          "open",
          [](S s, const c10::optional<OptionDict>& option) { s->open(option); })
      .def("close", [](S s) { s->close(); })
      .def(
          "write_audio_chunk",
          [](S s, int64_t i, const torch::Tensor& chunk) {
            s->write_audio_chunk(static_cast<int>(i), chunk);
          })
      .def(
          "write_video_chunk",
          [](S s, int64_t i, const torch::Tensor& chunk) {
            s->write_video_chunk(static_cast<int>(i), chunk);
          })
      .def("flush", [](S s) { s->flush(); });
}

} // namespace
} // namespace ffmpeg
} // namespace torchaudio
