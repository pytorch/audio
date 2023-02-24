#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/binding_utils.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/stream_writer.h>

namespace torchaudio {
namespace io {
namespace {

struct StreamWriterBinding : public StreamWriter,
                             public torch::CustomClassHolder {
  using StreamWriter::StreamWriter;
};

using S = const c10::intrusive_ptr<StreamWriterBinding>&;

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.class_<StreamWriterBinding>("ffmpeg_StreamWriter")
      .def(torch::init<>(
          [](const std::string& dst, const c10::optional<std::string>& format) {
            return c10::make_intrusive<StreamWriterBinding>(dst, format);
          }))
      .def(
          "add_audio_stream",
          [](S s,
             int64_t sample_rate,
             int64_t num_channels,
             const std::string& format,
             const c10::optional<std::string>& encoder,
             const c10::optional<OptionDictC10>& encoder_option,
             const c10::optional<std::string>& encoder_format) {
            s->add_audio_stream(
                sample_rate,
                num_channels,
                format,
                encoder,
                from_c10(encoder_option),
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
             const c10::optional<OptionDictC10>& encoder_option,
             const c10::optional<std::string>& encoder_format,
             const c10::optional<std::string>& hw_accel) {
            s->add_video_stream(
                frame_rate,
                width,
                height,
                format,
                encoder,
                from_c10(encoder_option),
                encoder_format,
                hw_accel);
          })
      .def(
          "set_metadata",
          [](S s, const OptionDictC10& metadata) {
            s->set_metadata(from_c10(metadata));
          })
      .def("dump_format", [](S s, int64_t i) { s->dump_format(i); })
      .def(
          "open",
          [](S s, const c10::optional<OptionDictC10>& option) {
            s->open(from_c10(option));
          })
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
} // namespace io
} // namespace torchaudio
