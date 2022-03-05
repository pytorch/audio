#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
namespace torchaudio {
namespace ffmpeg {

class FilterGraph {
  // Parameters required for `reset`
  // Recreats the underlying filter_graph struct
  AVRational input_time_base;
  AVCodecParameters* codecpar;
  std::string filter_description;

  // Constant just for convenient access.
  AVMediaType media_type;

  AVFilterGraphPtr pFilterGraph;
  // AVFilterContext is freed as a part of AVFilterGraph
  // so we do not manage the resource.
  AVFilterContext* buffersrc_ctx = nullptr;
  AVFilterContext* buffersink_ctx = nullptr;

 public:
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
  // Query method
  //////////////////////////////////////////////////////////////////////////////
  std::string get_description() const;

  //////////////////////////////////////////////////////////////////////////////
  // Configuration methods
  //////////////////////////////////////////////////////////////////////////////
  void init();

  void reset();

 private:
  void add_src();

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
