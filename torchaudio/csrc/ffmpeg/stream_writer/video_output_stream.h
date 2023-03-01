#pragma once
#include <torchaudio/csrc/ffmpeg/stream_writer/output_stream.h>
#include <torchaudio/csrc/ffmpeg/stream_writer/video_converter.h>

namespace torchaudio::io {

struct VideoOutputStream : OutputStream {
  VideoTensorConverter converter;
  AVBufferRefPtr hw_device_ctx;
  AVBufferRefPtr hw_frame_ctx;
  AVCodecContextPtr codec_ctx;

  VideoOutputStream(
      AVFormatContext* format_ctx,
      AVPixelFormat src_fmt,
      AVCodecContextPtr&& codec_ctx,
      AVBufferRefPtr&& hw_device_ctx,
      AVBufferRefPtr&& hw_frame_ctx);

  void write_chunk(const torch::Tensor& frames) override;

  ~VideoOutputStream() override = default;
};

} // namespace torchaudio::io
