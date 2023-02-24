#pragma once
#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader.h>

namespace torchaudio {
namespace io {

// Because TorchScript requires c10::Dict type to pass dict,
// while PyBind11 requires std::map type to pass dict,
// we duplicate the return tuple.
// Even though all the PyBind-based implementations are placed
// in `pybind` directory, because std::map does not require pybind11
// header, we define both of them here, for the sake of
// better locality/maintainability.

using SrcInfo = std::tuple<
    std::string, // media_type
    std::string, // codec name
    std::string, // codec long name
    std::string, // format name
    int64_t, // bit_rate
    int64_t, // num_frames
    int64_t, // bits_per_sample
    c10::Dict<std::string, std::string>, // metadata
    // Audio
    double, // sample_rate
    int64_t, // num_channels
    // Video
    int64_t, // width
    int64_t, // height
    double // frame_rate
    >;

using OutInfo = std::tuple<
    int64_t, // source index
    std::string // filter description
    >;

using ChunkData = std::tuple<torch::Tensor, double>;

// Structure to implement wrapper API around StreamReader, which is more
// suitable for Binding the code (i.e. it receives/returns pritimitves)
struct StreamReaderBinding : public StreamReader,
                             public torch::CustomClassHolder {
  using StreamReader::StreamReader;

  SrcInfo get_src_stream_info(int64_t i);
  OutInfo get_out_stream_info(int64_t i);

  std::vector<c10::optional<ChunkData>> pop_chunks();
};

} // namespace io
} // namespace torchaudio
