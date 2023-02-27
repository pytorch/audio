#pragma once
#include <torchaudio/csrc/ffmpeg/stream_writer/output_stream.h>

namespace torchaudio::io {

struct AudioOutputStream : OutputStream {
  AVFramePtr src_frame;
  int64_t frame_capacity;
  AVCodecContextPtr codec_ctx;

  AudioOutputStream(
      AVFormatContext* format_ctx,
      AVSampleFormat src_fmt,
      AVCodecContextPtr&& codec_ctx);

  void write_chunk(const torch::Tensor& waveform) override;
  ~AudioOutputStream() override = default;
};

} // namespace torchaudio::io
