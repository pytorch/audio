#pragma once
#include <torch/types.h>

namespace torchaudio::io {

using OptionDict = std::map<std::string, std::string>;
using OptionDictC10 = c10::Dict<std::string, std::string>;

OptionDictC10 to_c10(const OptionDict&);
OptionDict from_c10(const OptionDictC10&);
c10::optional<OptionDict> from_c10(const c10::optional<OptionDictC10>&);

} // namespace torchaudio::io
