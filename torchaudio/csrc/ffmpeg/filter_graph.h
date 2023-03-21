#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
namespace torchaudio {
namespace io {

/// Used to report the output formats of filter graph.
struct FilterGraphOutputInfo {
  AVMediaType type = AVMEDIA_TYPE_UNKNOWN;
  int format = -1;

  AVRational time_base = {1, 1};

  // Audio
  int sample_rate = -1;
  int num_channels = -1;

  // Video
  AVRational frame_rate = {0, 1};
  int height = -1;
  int width = -1;
};

class FilterGraph {
  AVMediaType media_type;

  AVFilterGraphPtr pFilterGraph;

  // AVFilterContext is freed as a part of AVFilterGraph
  // so we do not manage the resource.
  AVFilterContext* buffersrc_ctx = nullptr;
  AVFilterContext* buffersink_ctx = nullptr;

 public:
  explicit FilterGraph(AVMediaType media_type);
  // Custom destructor to release AVFilterGraph*
  ~FilterGraph() = default;
  // Non-copyable
  FilterGraph(const FilterGraph&) = delete;
  FilterGraph& operator=(const FilterGraph&) = delete;
  // Movable
  FilterGraph(FilterGraph&&) = default;
  FilterGraph& operator=(FilterGraph&&) = default;

  //////////////////////////////////////////////////////////////////////////////
  // Configuration methods
  //////////////////////////////////////////////////////////////////////////////
  void add_audio_src(
      AVSampleFormat format,
      AVRational time_base,
      int sample_rate,
      uint64_t channel_layout);

  void add_video_src(
      AVPixelFormat format,
      AVRational time_base,
      AVRational frame_rate,
      int width,
      int height,
      AVRational sample_aspect_ratio);

  void add_src(const std::string& arg);

  void add_sink();

  void add_process(const std::string& filter_description);

  void create_filter(AVBufferRef* hw_frames_ctx = nullptr);

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
  [[nodiscard]] FilterGraphOutputInfo get_output_info() const;

  //////////////////////////////////////////////////////////////////////////////
  // Streaming process
  //////////////////////////////////////////////////////////////////////////////
 public:
  int add_frame(AVFrame* pInputFrame);
  int get_frame(AVFrame* pOutputFrame);
};

} // namespace io
} // namespace torchaudio
