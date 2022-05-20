#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/stream_reader_wrapper.h>
#include <stdexcept>

namespace torchaudio {
namespace ffmpeg {

namespace {

OptionDict map(const c10::optional<c10::Dict<std::string, std::string>>& dict) {
  OptionDict ret;
  if (!dict.has_value()) {
    return ret;
  }
  for (const auto& it : dict.value()) {
    ret.insert({it.key(), it.value()});
  }
  return ret;
}

c10::intrusive_ptr<StreamReaderBinding> init(
    const std::string& src,
    const c10::optional<std::string>& device,
    const c10::optional<c10::Dict<std::string, std::string>>& option) {
  return c10::make_intrusive<StreamReaderBinding>(
      get_input_format_context(src, device, map(option)));
}

std::tuple<c10::optional<torch::Tensor>, int64_t> load(const std::string& src) {
  StreamReaderBinding s{get_input_format_context(src, {}, {})};
  int i = s.find_best_audio_stream();
  auto sinfo = s.Streamer::get_src_stream_info(i);
  int64_t sample_rate = static_cast<int64_t>(sinfo.sample_rate);
  s.add_audio_stream(i, -1, -1, {}, {}, {});
  s.process_all_packets();
  auto tensors = s.pop_chunks();
  return std::make_tuple<>(tensors[0], sample_rate);
}

using S = const c10::intrusive_ptr<StreamReaderBinding>&;

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::ffmpeg_init", []() {
    avdevice_register_all();
    if (av_log_get_level() == AV_LOG_INFO)
      av_log_set_level(AV_LOG_ERROR);
  });
  m.def("torchaudio::ffmpeg_load", load);
  m.class_<StreamReaderBinding>("ffmpeg_Streamer");
  m.def("torchaudio::ffmpeg_streamer_init", init);
  m.def("torchaudio::ffmpeg_streamer_num_src_streams", [](S s) {
    return s->num_src_streams();
  });
  m.def("torchaudio::ffmpeg_streamer_num_out_streams", [](S s) {
    return s->num_out_streams();
  });
  m.def("torchaudio::ffmpeg_streamer_get_src_stream_info", [](S s, int64_t i) {
    return s->get_src_stream_info(i);
  });
  m.def("torchaudio::ffmpeg_streamer_get_out_stream_info", [](S s, int64_t i) {
    return s->get_out_stream_info(i);
  });
  m.def("torchaudio::ffmpeg_streamer_find_best_audio_stream", [](S s) {
    return s->find_best_audio_stream();
  });
  m.def("torchaudio::ffmpeg_streamer_find_best_video_stream", [](S s) {
    return s->find_best_video_stream();
  });
  m.def("torchaudio::ffmpeg_streamer_seek", [](S s, double t) {
    return s->seek(t);
  });
  m.def(
      "torchaudio::ffmpeg_streamer_add_audio_stream",
      [](S s,
         int64_t i,
         int64_t frames_per_chunk,
         int64_t num_chunks,
         const c10::optional<std::string>& filter_desc,
         const c10::optional<std::string>& decoder,
         const c10::optional<c10::Dict<std::string, std::string>>&
             decoder_options) {
        s->add_audio_stream(
            i,
            frames_per_chunk,
            num_chunks,
            filter_desc,
            decoder,
            map(decoder_options));
      });
  m.def(
      "torchaudio::ffmpeg_streamer_add_video_stream",
      [](S s,
         int64_t i,
         int64_t frames_per_chunk,
         int64_t num_chunks,
         const c10::optional<std::string>& filter_desc,
         const c10::optional<std::string>& decoder,
         const c10::optional<c10::Dict<std::string, std::string>>&
             decoder_options,
         const c10::optional<std::string>& hw_accel) {
        s->add_video_stream(
            i,
            frames_per_chunk,
            num_chunks,
            filter_desc,
            decoder,
            map(decoder_options),
            hw_accel);
      });
  m.def("torchaudio::ffmpeg_streamer_remove_stream", [](S s, int64_t i) {
    s->remove_stream(i);
  });
  m.def(
      "torchaudio::ffmpeg_streamer_process_packet",
      [](S s, const c10::optional<double>& timeout, const double backoff) {
        return s->process_packet(timeout, backoff);
      });
  m.def("torchaudio::ffmpeg_streamer_process_all_packets", [](S s) {
    s->process_all_packets();
  });
  m.def("torchaudio::ffmpeg_streamer_is_buffer_ready", [](S s) {
    return s->is_buffer_ready();
  });
  m.def("torchaudio::ffmpeg_streamer_pop_chunks", [](S s) {
    return s->pop_chunks();
  });
}

} // namespace
} // namespace ffmpeg
} // namespace torchaudio
