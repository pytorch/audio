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
  Sink(
      AVRational input_time_base,
      AVCodecParameters* codecpar,
      int frames_per_chunk,
      int num_chunks,
      std::string filter_description);

  int process_frame(AVFrame* frame);
  bool is_buffer_ready() const;

  void flush();
};

} // namespace ffmpeg
} // namespace torchaudio
