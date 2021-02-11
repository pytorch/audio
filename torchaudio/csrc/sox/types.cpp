#include <torchaudio/csrc/sox/types.h>

namespace torchaudio {
namespace sox {

std::string to_string(Encoding v) {
  switch(v) {
  case Encoding::UNKNOWN:
    return "UNKNOWN";
  case Encoding::PCM_S:
    return "PCM_S";
  case Encoding::PCM_U:
    return "PCM_U";
  case Encoding::PCM_F:
    return "PCM_F";
  case Encoding::FLAC:
    return "FLAC";
  case Encoding::ULAW:
    return "ULAW";
  case Encoding::ALAW:
    return "ALAW";
  case Encoding::MP3:
    return "MP3";
  case Encoding::VORBIS:
    return "VORBIS";
  case Encoding::AMR_WB:
    return "AMR_WB";
  case Encoding::AMR_NB:
    return "AMR_NB";
  case Encoding::OPUS:
    return "OPUS";
  default:
    throw std::runtime_error("Internal Error: unexpected encoding.");
  }
}

Encoding from_string(const c10::optional<std::string>& encoding) {
  if (!encoding.has_value())
    return Encoding::NOT_PROVIDED;
  std::string v = encoding.get();
  if (v == "PCM_S")
    return Encoding::PCM_S;
  if (v == "PCM_U")
    return Encoding::PCM_U;
  if (v == "PCM_F")
    return Encoding::PCM_F;
  if (v == "ULAW")
    return Encoding::ULAW;
  if (v == "ALAW")
    return Encoding::ALAW;
  std::ostringstream stream;
  stream << "Internal Error: unexpected encoding value: " << v;
  throw std::runtime_error(stream.str());
}

} // namespace sox
} // namespace torchaudio
