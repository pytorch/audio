#pragma once
#include <torch/script.h>
#include <torchaudio/csrc/ffmpeg/streamer.h>

namespace torchaudio {
namespace ffmpeg {

using SrcInfo = std::tuple<
    std::string, // media_type
    std::string, // codec name
    std::string, // codec long name
    std::string, // format name
    int64_t, // bit_rate
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

// Structure to implement wrapper API around Streamer, which is more suitable
// for Binding the code (i.e. it receives/returns pritimitves)
struct StreamReaderBinding : public Streamer, public torch::CustomClassHolder {
  explicit StreamReaderBinding(AVFormatContextPtr&& p);
  SrcInfo get_src_stream_info(int64_t i);
  OutInfo get_out_stream_info(int64_t i);

  int64_t process_packet(
      const c10::optional<double>& timeout = c10::optional<double>(),
      const double backoff = 10.);

  void process_all_packets();
};

} // namespace ffmpeg
} // namespace torchaudio
