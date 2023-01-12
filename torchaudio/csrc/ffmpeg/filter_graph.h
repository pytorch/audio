#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
namespace torchaudio {
namespace ffmpeg {

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
      int width,
      int height,
      AVRational sample_aspect_ratio);

  void add_src(const std::string& arg);

  void add_sink();

  void add_process(const std::string& filter_description);

  void create_filter();

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
  [[nodiscard]] AVRational get_output_timebase() const;
  [[nodiscard]] int get_output_sample_rate() const;
  [[nodiscard]] int get_output_channels() const;

  //////////////////////////////////////////////////////////////////////////////
  // Streaming process
  //////////////////////////////////////////////////////////////////////////////
 public:
  int add_frame(AVFrame* pInputFrame);
  int get_frame(AVFrame* pOutputFrame);
};

} // namespace ffmpeg
} // namespace torchaudio
