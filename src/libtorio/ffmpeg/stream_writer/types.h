#pragma once
namespace torio::io {

struct CodecConfig {
  int bit_rate = -1;
  int compression_level = -1;

  // qscale corresponds to ffmpeg CLI's qscale.
  // Example: MP3
  // https://trac.ffmpeg.org/wiki/Encode/MP3
  // This should be set like
  // https://github.com/FFmpeg/FFmpeg/blob/n4.3.2/fftools/ffmpeg_opt.c#L1550
  const c10::optional<int> qscale = -1;

  // video
  int gop_size = -1;
  int max_b_frames = -1;
};
} // namespace torio::io
