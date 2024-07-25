#pragma once

#include <libtorio/ffmpeg/ffmpeg.h>
#include <torch/types.h>

namespace torio::io {

class TensorConverter {
 public:
  // Initialization is one-time process applied to frames before the iteration
  // starts. i.e. either convert to NHWC.
  using InitFunc = std::function<torch::Tensor(const torch::Tensor&, AVFrame*)>;
  // Convert function writes input frame Tensor to destinatoin AVFrame
  // both tensor input and AVFrame are expected to be valid and properly
  // allocated. (i.e. glorified copy). It is used in Iterator.
  using ConvertFunc = std::function<void(const torch::Tensor&, AVFrame*)>;

  //////////////////////////////////////////////////////////////////////////////
  // Generator
  //////////////////////////////////////////////////////////////////////////////
  // Generator class is responsible for implementing an interface
  // compatible with range-based for loop interface (begin and end).
  class Generator {
   public:
    ////////////////////////////////////////////////////////////////////////////
    // Iterator
    ////////////////////////////////////////////////////////////////////////////
    // Iterator class is responsible for implementing iterator protocol, that is
    // increment, comaprison against, and dereference (applying conversion
    // function in it).
    class Iterator {
      // Tensor to be sliced
      //  - audio: NC, CPU, uint8|int16|float|double
      //  - video: NCHW or NHWC, CPU or CUDA, uint8
      // It will be sliced at dereference time.
      const torch::Tensor frames;
      // Output buffer (not owned, but modified by Iterator)
      AVFrame* buffer;
      // Function that converts one frame Tensor into AVFrame.
      ConvertFunc& convert_func;

      // Index
      int64_t step;
      int64_t i = 0;

     public:
      Iterator(
          const torch::Tensor tensor,
          AVFrame* buffer,
          ConvertFunc& convert_func,
          int64_t step);

      Iterator& operator++();
      AVFrame* operator*() const;
      bool operator!=(const int64_t other) const;
    };

   private:
    // Input Tensor:
    //  - video: NCHW, CPU|CUDA, uint8,
    //  - audio: NC, CPU, uin8|int16|int32|in64|float32|double
    torch::Tensor frames;

    // Output buffer (not owned, passed to iterator)
    AVFrame* buffer;

    // ops: not owned.
    ConvertFunc& convert_func;

    int64_t step;

   public:
    Generator(
        torch::Tensor frames,
        AVFrame* buffer,
        ConvertFunc& convert_func,
        int64_t step = 1);

    [[nodiscard]] Iterator begin() const;
    [[nodiscard]] int64_t end() const;
  };

 private:
  AVFrame* buffer;
  const int buffer_size = 1;

  InitFunc init_func{};
  ConvertFunc convert_func{};

 public:
  TensorConverter(AVMediaType type, AVFrame* buffer, int buffer_size = 1);
  Generator convert(const torch::Tensor& t);
};

} // namespace torio::io
