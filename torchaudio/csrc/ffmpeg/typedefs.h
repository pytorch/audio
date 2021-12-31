#pragma once

#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <iostream>

namespace torchaudio {
namespace ffmpeg {

struct SrcStreamInfo {
  AVMediaType media_type;
  const char* codec_name = NULL;
  const char* codec_long_name = NULL;
  const char* fmt_name = NULL;
  int bit_rate = 0;
  // Audio
  double sample_rate = 0;
  int num_channels = 0;
  // Video
  int width = 0;
  int height = 0;
  double frame_rate = 0;
};

struct OutputStreamInfo {
  int source_index;
  std::string filter_description;
  double rate;
  OutputStreamInfo() = default;
};

} // namespace ffmpeg
} // namespace torchaudio
