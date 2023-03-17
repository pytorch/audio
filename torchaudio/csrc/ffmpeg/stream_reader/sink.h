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
  AVCodecContext* codec_ctx;
  AVRational frame_rate;
  std::string filter_description;
  FilterGraph filter;
  // time_base of filter graph output, used for PTS calc
  AVRational output_time_base;

 public:
  std::unique_ptr<Buffer> buffer;
  Sink(
      AVRational input_time_base,
      AVCodecContext* codec_ctx,
      int frames_per_chunk,
      int num_chunks,
      AVRational frame_rate,
      const c10::optional<std::string>& filter_description,
      const torch::Device& device);

  [[nodiscard]] std::string get_filter_description() const;
  [[nodiscard]] FilterGraphOutputInfo get_filter_output_info() const;

  int process_frame(AVFrame* frame);
  bool is_buffer_ready() const;

  void flush();
};

} // namespace io
} // namespace torchaudio
