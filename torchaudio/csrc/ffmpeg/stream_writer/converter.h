#pragma once

#include <torch/types.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>

namespace torchaudio::io {

//////////////////////////////////////////////////////////////////////////////
// Generator
//////////////////////////////////////////////////////////////////////////////
// Genrator class is responsible for implementing an interface compatible with
// range-based for loop interface (begin and end), and initialization of frame
// data (channel reordering and ensuring the contiguous-ness).
class Generator {
 public:
  // Convert function writes input frame Tensor to destinatoin AVFrame
  // both tensor input and AVFrame are expected to be valid and properly
  // allocated. (i.e. glorified copy)
  // It is one-to-one conversion. Performed in Iterator.
  using ConvertFunc = std::function<void(const torch::Tensor&, AVFrame*)>;

  ////////////////////////////////////////////////////////////////////////////
  // Iterator
  ////////////////////////////////////////////////////////////////////////////
  // Iterator class is responsible for implementing iterator protocol, that is
  // increment, comaprison against, and dereference (applying conversion
  // function in it).
  class Iterator {
    // Input tensor, has to be NCHW or NHWC, uint8, CPU or CUDA
    // It will be sliced at dereference time.
    const torch::Tensor frames;
    // Output buffer (not owned, but modified by Iterator)
    AVFrame* buffer;
    // Function that converts one frame Tensor into AVFrame.
    ConvertFunc& convert_func;

    // Index
    int64_t i = 0;

   public:
    Iterator(
        const torch::Tensor tensor,
        AVFrame* buffer,
        ConvertFunc& convert_func);

    Iterator& operator++();
    AVFrame* operator*() const;
    bool operator!=(const int64_t other) const;
  };

 private:
  // Tensor representing video frames provided by client code
  // Expected (and validated) to be NCHW, uint8.
  torch::Tensor frames;

  // Output buffer (not owned, passed to iterator)
  AVFrame* buffer;

  // ops: not owned.
  ConvertFunc& convert_func;

 public:
  Generator(torch::Tensor frames, AVFrame* buffer, ConvertFunc& convert_func);

  [[nodiscard]] Iterator begin() const;
  [[nodiscard]] int64_t end() const;
};
} // namespace torchaudio::io
