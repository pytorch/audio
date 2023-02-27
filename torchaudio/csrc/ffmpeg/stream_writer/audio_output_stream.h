#pragma once
#include <torchaudio/csrc/ffmpeg/stream_writer/output_stream.h>

namespace torchaudio::io {

struct AudioOutputStream : OutputStream {
  int64_t frame_capacity;

  AudioOutputStream(
      AVFormatContext* format_ctx,
      AVCodecContextPtr&& codec_ctx,
      std::unique_ptr<FilterGraph>&& filter,
      AVFramePtr&& src_frame,
      int64_t frame_capacity);

  void write_chunk(const torch::Tensor& waveform) override;
  ~AudioOutputStream() override = default;
};

} // namespace torchaudio::io
