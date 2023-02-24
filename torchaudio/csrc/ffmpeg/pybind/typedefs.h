#pragma once
#include <torch/extension.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio {
namespace io {

struct FileObj {
  py::object fileobj;
  int buffer_size;
  AVIOContextPtr pAVIO;
  FileObj(py::object fileobj, int buffer_size, bool writable);
};

} // namespace io
} // namespace torchaudio
