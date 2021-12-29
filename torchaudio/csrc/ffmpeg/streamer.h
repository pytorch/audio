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
      const std::map<std::string, std::string>& option);
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
  // Check if all the buffers of the output streams are ready.
  bool is_buffer_ready() const;

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
  void seek(double timestamp);

  void add_audio_stream(
      int i,
      int frames_per_chunk,
      int num_chunks,
      double rate,
      std::string filter_desc);
  void add_video_stream(
      int i,
      int frames_per_chunk,
      int num_chunks,
      double rate,
      std::string filter_desc);
  void remove_stream(int i);

 private:
  void add_stream(
      int i,
      AVMediaType media_type,
      int frames_per_chunk,
      int num_chunks,
      double rate,
      std::string filter_desc);

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
  std::vector<c10::optional<torch::Tensor>> pop_chunks();
};

} // namespace ffmpeg
} // namespace torchaudio
