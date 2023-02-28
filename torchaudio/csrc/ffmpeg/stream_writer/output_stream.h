#pragma once

#include <torch/types.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/encoder.h>

namespace torchaudio::io {

struct OutputStream {
  // Reference to codec context
  AVCodecContext* codec_ctx;
  // Encoder + Muxer
  Encoder encoder;
  // Filter for additional processing
  FilterGraph filter;
  // frame that output from FilterGraph is written
  AVFramePtr dst_frame;
  // The number of samples written so far
  int64_t num_frames;

  OutputStream(
      AVFormatContext* format_ctx,
      AVCodecContext* codec_ctx,
      FilterGraph&& filter);

  virtual void write_chunk(const torch::Tensor& input) = 0;
  void process_frame(AVFrame* src);
  void flush();
  virtual ~OutputStream() = default;
};

} // namespace torchaudio::io
