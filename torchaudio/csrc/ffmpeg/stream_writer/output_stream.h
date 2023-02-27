#pragma once

#include <torch/types.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>

namespace torchaudio::io {

struct OutputStream {
  // Reference to the AVFormatContext that this stream belongs to
  AVFormatContext* format_ctx;
  // Stream object that OutputStream is responsible for managing
  AVStream* stream;
  // Codec context (encoder)
  AVCodecContextPtr codec_ctx;
  // Filter for additional processing
  std::unique_ptr<FilterGraph> filter;
  // frame that user-provided input data is written
  AVFramePtr src_frame;
  // frame that output from FilterGraph is written
  AVFramePtr dst_frame;
  // The number of samples written so far
  int64_t num_frames;
  // Temporary object used during the encoding
  AVPacketPtr packet;

  OutputStream(
      AVFormatContext* format_ctx,
      AVStream* stream,
      AVCodecContextPtr&& codec_ctx,
      std::unique_ptr<FilterGraph>&& filter,
      AVFramePtr&& src_frame);

  virtual void write_chunk(const torch::Tensor& input) = 0;
  void process_frame(AVFrame* src);
  void flush();
  virtual ~OutputStream() = default;
};

struct AudioOutputStream : OutputStream {
  int64_t frame_capacity;

  AudioOutputStream(
      AVFormatContext* format_ctx,
      AVStream* stream,
      AVCodecContextPtr&& codec_ctx,
      std::unique_ptr<FilterGraph>&& filter,
      AVFramePtr&& src_frame,
      int64_t frame_capacity);

  void write_chunk(const torch::Tensor& waveform) override;
  ~AudioOutputStream() override = default;
};

struct VideoOutputStream : OutputStream {
  // Video-only: HW acceleration
  AVBufferRefPtr hw_device_ctx;
  AVBufferRefPtr hw_frame_ctx;

  VideoOutputStream(
      AVFormatContext* format_ctx,
      AVStream* stream,
      AVCodecContextPtr&& codec_ctx,
      std::unique_ptr<FilterGraph>&& filter,
      AVFramePtr&& src_frame,
      AVBufferRefPtr&& hw_device_ctx,
      AVBufferRefPtr&& hw_frame_ctx);

  void write_chunk(const torch::Tensor& frames) override;
  ~VideoOutputStream() override = default;
};

} // namespace torchaudio::io
