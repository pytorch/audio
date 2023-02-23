#include <torchaudio/csrc/ffmpeg/binding_utils.h>

namespace torchaudio::io {

OptionDictC10 to_c10(const OptionDict& src) {
  OptionDictC10 ret;
  for (auto const& [key, value] : src) {
    ret.insert(key, value);
  }
  return ret;
}

OptionDict from_c10(const OptionDictC10& src) {
  OptionDict ret;
  for (const auto& it : src) {
    ret.emplace(it.key(), it.value());
  }
  return ret;
}

c10::optional<OptionDict> from_c10(const c10::optional<OptionDictC10>& src) {
  if (src) {
    return {from_c10(src.value())};
  }
  return {};
}

} // namespace torchaudio::io
