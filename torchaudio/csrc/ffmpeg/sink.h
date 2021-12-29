#pragma once

#include <torchaudio/csrc/ffmpeg/buffer.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>

namespace torchaudio {
namespace ffmpeg {

class Sink {
  AVFramePtr frame;

 public:
  FilterGraph filter;
  std::unique_ptr<Buffer> buffer;
  double time_base;
  Sink(
      AVRational input_time_base,
      AVCodecParameters* codecpar,
      int frames_per_chunk,
      int num_chunks,
      double output_time_base,
      std::string filter_description);

  int process_frame(AVFrame* frame);
  bool is_buffer_ready() const;
};

} // namespace ffmpeg
} // namespace torchaudio
