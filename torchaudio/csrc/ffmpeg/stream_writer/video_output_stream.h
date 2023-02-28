#pragma once
#include <torchaudio/csrc/ffmpeg/stream_writer/output_stream.h>

namespace torchaudio::io {

struct VideoOutputStream : OutputStream {
  AVFramePtr src_frame;

  AVBufferRefPtr hw_device_ctx;
  AVBufferRefPtr hw_frame_ctx;
  AVCodecContextPtr codec_ctx;

  VideoOutputStream(
      AVFormatContext* format_ctx,
      AVPixelFormat src_fmt,
      AVCodecContextPtr&& codec_ctx,
      AVBufferRefPtr&& hw_device_ctx,
      AVBufferRefPtr&& hw_frame_ctx,
      const torch::Device& device);

  void write_chunk(const torch::Tensor& frames) override;
  void process_frame();

  ~VideoOutputStream() override = default;
};

} // namespace torchaudio::io
