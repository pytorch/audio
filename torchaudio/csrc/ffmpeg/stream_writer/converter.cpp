#include <torchaudio/csrc/ffmpeg/stream_writer/converter.h>

namespace torchaudio::io {

using Iterator = Generator::Iterator;
using ConvertFunc = Generator::ConvertFunc;

////////////////////////////////////////////////////////////////////////////////
// Generator
////////////////////////////////////////////////////////////////////////////////

Generator::Generator(torch::Tensor frames_, AVFrame* buff, ConvertFunc& func)
    : frames(std::move(frames_)), buffer(buff), convert_func(func) {}

Iterator Generator::begin() const {
  return Iterator{frames, buffer, convert_func};
}

int64_t Generator::end() const {
  return frames.size(0);
}

////////////////////////////////////////////////////////////////////////////////
// Iterator
////////////////////////////////////////////////////////////////////////////////

Iterator::Iterator(
    const torch::Tensor frames_,
    AVFrame* buffer_,
    ConvertFunc& convert_func_)
    : frames(frames_), buffer(buffer_), convert_func(convert_func_) {}

Iterator& Iterator::operator++() {
  ++i;
  return *this;
}

AVFrame* Iterator::operator*() const {
  convert_func(frames.index({i}), buffer);
  return buffer;
}

bool Iterator::operator!=(const int64_t other) const {
  return i != other;
}

} // namespace torchaudio::io
