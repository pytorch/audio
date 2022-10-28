#pragma once

#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader_wrapper.h>

namespace torchaudio {
namespace ffmpeg {

// Helper structure to keep track of until where the decoding has happened
struct TensorIndexer {
  torch::Tensor src;
  size_t index = 0;
  const uint8_t* data;
  const size_t numel;
  AVIOContextPtr pAVIO;

  TensorIndexer(const torch::Tensor& src, int buffer_size);
};

// Structure to implement wrapper API around StreamReader, which is more
// suitable for Binding the code (i.e. it receives/returns pritimitves)
struct StreamReaderTensorBinding : protected TensorIndexer,
                                   public StreamReaderBinding {
  StreamReaderTensorBinding(
      const torch::Tensor& src,
      const c10::optional<std::string>& device,
      const c10::optional<OptionDict>& option,
      int buffer_size);
};

} // namespace ffmpeg
} // namespace torchaudio
