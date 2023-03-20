#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/buffer.h>

namespace torchaudio {
namespace io {

class Sink {
  AVFramePtr frame{};

  // Parameters for recreating FilterGraph
  AVRational input_time_base;
  AVCodecContext* codec_ctx;
  AVRational frame_rate;

 public:
  const std::string filter_description;
  FilterGraph filter;
  std::unique_ptr<Buffer> buffer;

  Sink(
      AVRational input_time_base,
      AVCodecContext* codec_ctx,
      int frames_per_chunk,
      int num_chunks,
      AVRational frame_rate,
      const std::string& filter_description,
      const torch::Device& device);

  int process_frame(AVFrame* frame);
  bool is_buffer_ready() const;

  void flush();
};

} // namespace io
} // namespace torchaudio
