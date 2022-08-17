#pragma once
#include <torch/extension.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {

struct FileObj {
  py::object fileobj;
  int buffer_size;
  AVIOContextPtr pAVIO;
  FileObj(py::object fileobj, int buffer_size, bool writable);
};

using OptionMap = std::map<std::string, std::string>;

OptionDict map2dict(const OptionMap& src);

c10::optional<OptionDict> map2dict(const c10::optional<OptionMap>& src);

OptionMap dict2map(const OptionDict& src);

} // namespace ffmpeg
} // namespace torchaudio
