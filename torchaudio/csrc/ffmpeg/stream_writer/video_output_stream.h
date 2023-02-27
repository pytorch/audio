#pragma once
#include <torchaudio/csrc/ffmpeg/stream_writer/output_stream.h>

namespace torchaudio::io {

struct VideoOutputStream : OutputStream {
  AVBufferRefPtr hw_device_ctx;
  AVBufferRefPtr hw_frame_ctx;

  VideoOutputStream(
      AVFormatContext* format_ctx,
      AVCodecContextPtr&& codec_ctx,
      std::unique_ptr<FilterGraph>&& filter,
      AVFramePtr&& src_frame,
      AVBufferRefPtr&& hw_device_ctx,
      AVBufferRefPtr&& hw_frame_ctx);

  void write_chunk(const torch::Tensor& frames) override;
  ~VideoOutputStream() override = default;
};

} // namespace torchaudio::io
