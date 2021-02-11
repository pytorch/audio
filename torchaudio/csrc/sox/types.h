#ifndef TORCHAUDIO_SOX_TYPES_H
#define TORCHAUDIO_SOX_TYPES_H

#include <torch/script.h>

namespace torchaudio {
namespace sox {

enum class Encoding {
  NOT_PROVIDED,
  UNKNOWN,
  PCM_S,
  PCM_U,
  PCM_F,
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
Encoding from_string(const c10::optional<std::string>& encoding);

} // namespace sox
} // namespace torchaudio

#endif
