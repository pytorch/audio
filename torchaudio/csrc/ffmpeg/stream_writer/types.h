#pragma once
namespace torchaudio::io {

struct CodecConfig {
  int bit_rate = -1;
  int compression_level = -1;

  // video
  int gop_size = -1;
  int max_b_frames = -1;
};
} // namespace torchaudio::io
