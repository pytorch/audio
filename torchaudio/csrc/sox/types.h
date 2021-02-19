#ifndef TORCHAUDIO_SOX_TYPES_H
#define TORCHAUDIO_SOX_TYPES_H

#include <sox.h>
#include <torch/script.h>

namespace torchaudio {
namespace sox_utils {

enum class Format {
  WAV,
  MP3,
  FLAC,
  VORBIS,
  AMR_NB,
  AMR_WB,
  AMB,
  SPHERE,
  GSM,
};

Format get_format_from_string(const std::string& format);

enum class Encoding {
  NOT_PROVIDED,
  UNKNOWN,
  PCM_SIGNED,
  PCM_UNSIGNED,
  PCM_FLOAT,
  FLAC,
  ULAW,
  ALAW,
  MP3,
  VORBIS,
  AMR_WB,
  AMR_NB,
  OPUS,
};

std::string to_string(Encoding v);
Encoding get_encoding_from_option(const c10::optional<std::string> encoding);

enum class BitDepth : unsigned {
  NOT_PROVIDED = 0,
  B8 = 8,
  B16 = 16,
  B24 = 24,
  B32 = 32,
  B64 = 64,
};

BitDepth get_bit_depth_from_option(const c10::optional<int64_t> bit_depth);

std::string get_encoding(sox_encoding_t encoding);

} // namespace sox_utils
} // namespace torchaudio

#endif
