#include <torchaudio/csrc/ffmpeg/stream_writer/converter.h>

namespace torchaudio::io {

using Iterator = SlicingTensorConverter::Iterator;
using ConvertFunc = SlicingTensorConverter::ConvertFunc;

////////////////////////////////////////////////////////////////////////////////
// SlicingTensorConverter
////////////////////////////////////////////////////////////////////////////////

SlicingTensorConverter::SlicingTensorConverter(
    torch::Tensor frames_,
    AVFrame* buff,
    ConvertFunc& func,
    int64_t step_)
    : frames(std::move(frames_)),
      buffer(buff),
      convert_func(func),
      step(step_) {}

Iterator SlicingTensorConverter::begin() const {
  return Iterator{frames, buffer, convert_func, step};
}

int64_t SlicingTensorConverter::end() const {
  return frames.size(0);
}

////////////////////////////////////////////////////////////////////////////////
// Iterator
////////////////////////////////////////////////////////////////////////////////

Iterator::Iterator(
    const torch::Tensor frames_,
    AVFrame* buffer_,
    ConvertFunc& convert_func_,
    int64_t step_)
    : frames(frames_),
      buffer(buffer_),
      convert_func(convert_func_),
      step(step_) {}

Iterator& Iterator::operator++() {
  i += step;
  return *this;
}

AVFrame* Iterator::operator*() const {
  using namespace torch::indexing;
  convert_func(frames.index({Slice{i, i + step}}), buffer);
  return buffer;
}

bool Iterator::operator!=(const int64_t end) const {
  // This is used for detecting the end of iteraton.
  // For audio, iteration is done by
  return i < end;
}

} // namespace torchaudio::io
