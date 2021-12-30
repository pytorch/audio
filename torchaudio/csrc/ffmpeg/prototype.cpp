#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/streamer.h>

namespace torchaudio {
namespace ffmpeg {

namespace {

std::map<std::string, std::string> convert_dict(
    const c10::optional<c10::Dict<std::string, std::string>>& option) {
  std::map<std::string, std::string> opts;
  if (option) {
    for (auto& it : option.value()) {
      opts[it.key()] = it.value();
    }
  }
  return opts;
}

struct StreamerHolder : torch::CustomClassHolder {
  Streamer s;
  StreamerHolder(
      const std::string& src,
      c10::optional<std::string> device,
      c10::optional<c10::Dict<std::string, std::string>> option)
      : s(src, device.value_or(""), convert_dict(option)) {}
};

using S = c10::intrusive_ptr<StreamerHolder>;

S init(
    const std::string& src,
    c10::optional<std::string> device,
    c10::optional<c10::Dict<std::string, std::string>> option) {
  return c10::make_intrusive<StreamerHolder>(src, device, option);
}

using SrcInfo = std::tuple<
    std::string, // media_type
    std::string, // codec name
    std::string, // codec long name
    std::string, // format name
    int64_t, // bit_rate
    // Audio
    double, // sample_rate
    int64_t, // num_channels
    // Video
    int64_t, // width
    int64_t, // height
    double // frame_rate
    >;

SrcInfo convert(SrcStreamInfo ssi) {
  return SrcInfo(std::forward_as_tuple(
      av_get_media_type_string(ssi.media_type),
      ssi.codec_name,
      ssi.codec_long_name,
      ssi.fmt_name,
      ssi.bit_rate,
      ssi.sample_rate,
      ssi.num_channels,
      ssi.width,
      ssi.height,
      ssi.frame_rate));
}

SrcInfo get_src_stream_info(S s, int64_t i) {
  return convert(s->s.get_src_stream_info(i));
}

using OutInfo = std::tuple<
    int64_t, // source index
    std::string // filter description
    >;

OutInfo convert(OutputStreamInfo osi) {
  return OutInfo(
      std::forward_as_tuple(osi.source_index, osi.filter_description));
}

OutInfo get_out_stream_info(S s, int64_t i) {
  return convert(s->s.get_out_stream_info(i));
}

int64_t num_src_streams(S s) {
  return s->s.num_src_streams();
}

int64_t num_out_streams(S s) {
  return s->s.num_out_streams();
}

int64_t find_best_audio_stream(S s) {
  return s->s.find_best_audio_stream();
}

int64_t find_best_video_stream(S s) {
  return s->s.find_best_video_stream();
}

template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
  char buffer[512];
  std::snprintf(buffer, sizeof(buffer), format.c_str(), args...);
  return std::string(buffer);
}

std::string join(
    const std::vector<std::string>& components,
    const std::string& delim) {
  std::ostringstream s;
  for (int i = 0; i < components.size(); ++i) {
    if (i)
      s << delim;
    s << components[i];
  }
  return s.str();
}
std::string get_afilter_desc(
    const c10::optional<int64_t>& sample_rate,
    const c10::optional<c10::ScalarType>& dtype) {
  std::vector<std::string> components;
  if (sample_rate) {
    // TODO: test float sample rate
    components.emplace_back(
        string_format("aresample=%d", static_cast<int>(sample_rate.value())));
  }
  if (dtype) {
    AVSampleFormat fmt = [&]() {
      switch (dtype.value()) {
        case c10::ScalarType::Byte:
          return AV_SAMPLE_FMT_U8P;
        case c10::ScalarType::Short:
          return AV_SAMPLE_FMT_S16P;
        case c10::ScalarType::Int:
          return AV_SAMPLE_FMT_S32P;
        case c10::ScalarType::Long:
          return AV_SAMPLE_FMT_S64P;
        case c10::ScalarType::Float:
          return AV_SAMPLE_FMT_FLTP;
        case c10::ScalarType::Double:
          return AV_SAMPLE_FMT_DBLP;
        default:
          throw std::runtime_error("Unexpected dtype.");
      }
    }();
    components.emplace_back(
        string_format("aformat=sample_fmts=%s", av_get_sample_fmt_name(fmt)));
  }
  return join(components, ",");
}
std::string get_vfilter_desc(
    const c10::optional<double>& frame_rate,
    const c10::optional<int64_t>& width,
    const c10::optional<int64_t>& height,
    const c10::optional<std::string>& format) {
  // TODO:
  // - Add `flags` for different scale algorithm
  //   https://ffmpeg.org/ffmpeg-filters.html#scale
  // - Consider `framerate` as well
  //   https://ffmpeg.org/ffmpeg-filters.html#framerate

  // - scale
  //   https://ffmpeg.org/ffmpeg-filters.html#scale-1
  //   https://ffmpeg.org/ffmpeg-scaler.html#toc-Scaler-Options
  // - framerate
  //   https://ffmpeg.org/ffmpeg-filters.html#framerate

  // TODO:
  // - format
  //   https://ffmpeg.org/ffmpeg-filters.html#toc-format-1
  // - fps
  //   https://ffmpeg.org/ffmpeg-filters.html#fps-1
  std::vector<std::string> components;
  if (frame_rate)
    components.emplace_back(string_format("fps=%lf", frame_rate.value()));

  std::vector<std::string> scale_components;
  if (width)
    scale_components.emplace_back(string_format("width=%d", width.value()));
  if (height)
    scale_components.emplace_back(string_format("height=%d", height.value()));
  if (scale_components.size())
    components.emplace_back(
        string_format("scale=%s", join(scale_components, ":").c_str()));
  if (format) {
    // TODO:
    // Check other useful formats
    // https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
    AVPixelFormat fmt = [&]() {
      std::string val = format.value();
      if (val == "RGB")
        return AV_PIX_FMT_RGB24;
      if (val == "BGR")
        return AV_PIX_FMT_BGR24;
      if (val == "GRAY")
        return AV_PIX_FMT_GRAY8;
      throw std::runtime_error("Unexpected format: " + val);
    }();
    components.emplace_back(
        string_format("format=pix_fmts=%s", av_get_pix_fmt_name(fmt)));
  }
  return join(components, ",");
};

void add_basic_audio_stream(
    S s,
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<int64_t>& sample_rate,
    const c10::optional<c10::ScalarType>& dtype) {
  std::string filter_desc = get_afilter_desc(sample_rate, dtype);
  s->s.add_audio_stream(
      i, frames_per_chunk, num_chunks, sample_rate.value_or(-1), filter_desc);
}

void add_basic_video_stream(
    S s,
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<double>& frame_rate,
    const c10::optional<int64_t>& width,
    const c10::optional<int64_t>& height,
    const c10::optional<std::string>& format) {
  std::string filter_desc = get_vfilter_desc(frame_rate, width, height, format);
  s->s.add_video_stream(
      i, frames_per_chunk, num_chunks, frame_rate.value_or(-1), filter_desc);
}

void add_audio_stream(
    S s,
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<double>& sample_rate,
    const c10::optional<std::string>& filter_desc) {
  s->s.add_audio_stream(
      i,
      frames_per_chunk,
      num_chunks,
      sample_rate.value_or(-1),
      filter_desc.value_or(""));
}

void add_video_stream(
    S s,
    int64_t i,
    int64_t frames_per_chunk,
    int64_t num_chunks,
    const c10::optional<double>& frame_rate,
    const c10::optional<std::string>& filter_desc) {
  s->s.add_video_stream(
      i,
      frames_per_chunk,
      num_chunks,
      frame_rate.value_or(-1),
      filter_desc.value_or(""));
}

void remove_stream(S s, int64_t i) {
  s->s.remove_stream(i);
}

int64_t process_packet(S s) {
  return s->s.process_packet();
}

int64_t process_all_packets(S s) {
  return s->s.process_all_packets();
}

bool is_buffer_ready(S s) {
  return s->s.is_buffer_ready();
}

std::vector<c10::optional<torch::Tensor>> pop_chunks(S s) {
  return s->s.pop_chunks();
}

std::tuple<c10::optional<torch::Tensor>, int64_t> load(const std::string& src) {
  Streamer s{src, "", {}};
  int i = s.find_best_audio_stream();
  auto sinfo = s.get_src_stream_info(i);
  int64_t sample_rate = static_cast<int64_t>(sinfo.sample_rate);
  s.add_audio_stream(i, -1, -1, 1, "");
  s.process_all_packets();
  auto tensors = s.pop_chunks();
  return std::make_tuple<>(tensors[0], sample_rate);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::ffmpeg_init", []() {
    avdevice_register_all();
    if (av_log_get_level() == AV_LOG_INFO)
      av_log_set_level(AV_LOG_ERROR);
  });
  m.def("torchaudio::ffmpeg_load", load);
  m.class_<StreamerHolder>("ffmpeg_Streamer");
  m.def("torchaudio::ffmpeg_streamer_init", init);
  m.def("torchaudio::ffmpeg_streamer_num_src_streams", num_src_streams);
  m.def("torchaudio::ffmpeg_streamer_num_out_streams", num_out_streams);
  m.def("torchaudio::ffmpeg_streamer_get_src_stream_info", get_src_stream_info);
  m.def("torchaudio::ffmpeg_streamer_get_out_stream_info", get_out_stream_info);
  m.def(
      "torchaudio::ffmpeg_streamer_find_best_audio_stream",
      find_best_audio_stream);
  m.def(
      "torchaudio::ffmpeg_streamer_find_best_video_stream",
      find_best_video_stream);
  m.def(
      "torchaudio::ffmpeg_streamer_add_basic_audio_stream",
      add_basic_audio_stream);
  m.def(
      "torchaudio::ffmpeg_streamer_add_basic_video_stream",
      add_basic_video_stream);
  m.def("torchaudio::ffmpeg_streamer_add_audio_stream", add_audio_stream);
  m.def("torchaudio::ffmpeg_streamer_add_video_stream", add_video_stream);
  m.def("torchaudio::ffmpeg_streamer_remove_stream", remove_stream);
  m.def("torchaudio::ffmpeg_streamer_process_packet", process_packet);
  m.def("torchaudio::ffmpeg_streamer_process_all_packets", process_all_packets);
  m.def("torchaudio::ffmpeg_streamer_is_buffer_ready", is_buffer_ready);
  m.def("torchaudio::ffmpeg_streamer_pop_chunks", pop_chunks);
}

} // namespace
} // namespace ffmpeg
} // namespace torchaudio
