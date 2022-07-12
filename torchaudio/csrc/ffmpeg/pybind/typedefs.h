#pragma once
#include <torch/extension.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace ffmpeg {

struct FileObj {
  py::object fileobj;
  int buffer_size;
  AVIOContextPtr pAVIO;
  FileObj(py::object fileobj, int buffer_size);
};

c10::optional<OptionDict> map2dict(
    const c10::optional<std::map<std::string, std::string>>& src);

std::map<std::string, std::string> dict2map(const OptionDict& src);

} // namespace ffmpeg
} // namespace torchaudio
