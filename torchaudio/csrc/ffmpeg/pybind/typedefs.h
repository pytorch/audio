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

} // namespace ffmpeg
} // namespace torchaudio
