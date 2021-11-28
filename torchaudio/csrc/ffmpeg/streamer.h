#pragma once
#include <torchaudio/csrc/ffmpeg/decoder.h>
#include <torchaudio/csrc/ffmpeg/filter_graph.h>
#include <torchaudio/csrc/ffmpeg/stream_processor.h>
#include <torchaudio/csrc/ffmpeg/typedefs.h>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

class Streamer {
  AVFormatContextPtr pFormatContext;
  AVPacketPtr pPacket;

  std::vector<std::unique_ptr<StreamProcessor>> processors;
  // Mapping from user-facing stream index to internal index.
  // The first one is processor index,
  // the second is the map key inside of processor.
  std::vector<std::pair<int, int>> stream_indices;

 public:
  // Open the input and allocate the resource
  Streamer(
      const std::string& src,
      const std::string& device,
      AVDictionary** option);
  ~Streamer() = default;
  // Non-copyable
  Streamer(const Streamer&) = delete;
  Streamer& operator=(const Streamer&) = delete;
  // Movable
  Streamer(Streamer&&) = default;
  Streamer& operator=(Streamer&&) = default;

  //////////////////////////////////////////////////////////////////////////////
  // Helper methods
  //////////////////////////////////////////////////////////////////////////////
 private:
  void validate_open_stream() const;
  void validate_src_stream_index(int i) const;
  void validate_output_stream_index(int i) const;
  void validate_src_stream_type(int i, AVMediaType type);

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  // Find a suitable audio/video streams using heuristics from ffmpeg
  int find_best_audio_stream() const;
  int find_best_video_stream() const;
  // Fetch information about source streams
  int num_src_streams() const;
  SrcStreamInfo get_src_stream_info(int i) const;
  // Fetch information about output streams
  int num_out_streams() const;
  OutputStreamInfo get_out_stream_info(int i) const;

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
  // Enable stream - with filtering
  // TODO add necessary parameters
  void add_basic_audio_stream(int i, int sample_rate, AVSampleFormat fmt);
  void add_basic_video_stream(
      int i,
      int width,
      int height,
      double frame_rate,
      AVPixelFormat fmt);
  void add_custom_audio_stream(
      int i,
      const std::string& filter_desc,
      double rate);
  void add_custom_video_stream(
      int i,
      const std::string& filter_desc,
      double rate);
  void remove_stream(int i);

 private:
  void add_custom_stream(
      int i,
      AVMediaType media_type,
      const std::string& filter_desc,
      double rate);

 public:
  //////////////////////////////////////////////////////////////////////////////
  // Stream methods
  //////////////////////////////////////////////////////////////////////////////
  int process_packet();
  int process_all_packets();
  int drain();

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
  std::vector<torch::Tensor> get_chunks();
};

} // namespace ffmpeg
} // namespace torchaudio
