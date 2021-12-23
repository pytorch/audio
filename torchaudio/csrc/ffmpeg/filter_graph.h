#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
namespace torchaudio {
namespace ffmpeg {

class FilterGraph {
  AVMediaType media_type = AVMEDIA_TYPE_UNKNOWN;
  AVFilterGraphPtr pFilterGraph;
  // AVFilterContext is freed as a part of AVFilterGraph
  // so we do not manage the resource.
  AVFilterContext* buffersrc_ctx = nullptr;
  AVFilterContext* buffersink_ctx = nullptr;

 public:
  const std::string filter_description;

  FilterGraph(
      AVRational time_base,
      AVCodecParameters* codecpar,
      std::string filter_desc);
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
 private:
  void add_src(AVRational time_base, AVCodecParameters* codecpar);

  void add_sink();

  void add_process();

  void create_filter();

  //////////////////////////////////////////////////////////////////////////////
  // Streaming process
  //////////////////////////////////////////////////////////////////////////////
 public:
  int add_frame(AVFrame* pInputFrame);
  int get_frame(AVFrame* pOutputFrame);
};

} // namespace ffmpeg
} // namespace torchaudio
