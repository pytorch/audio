#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer.h>

namespace torchaudio {
namespace io {

class Sink {
  AVFramePtr frame;

  // Parameters for recreating FilterGraph
  AVRational input_time_base;
  AVCodecParameters* codecpar;
  std::string filter_description;
  std::unique_ptr<FilterGraph> filter;
  // time_base of filter graph output, used for PTS calc
  AVRational output_time_base;

 public:
  std::unique_ptr<Buffer> buffer;
  Sink(
      AVRational input_time_base,
      AVCodecParameters* codecpar,
      int frames_per_chunk,
      int num_chunks,
      const c10::optional<std::string>& filter_description,
      const torch::Device& device);

  std::string get_filter_description() const;
  int process_frame(AVFrame* frame);
  bool is_buffer_ready() const;

  void flush();
};

} // namespace io
} // namespace torchaudio
