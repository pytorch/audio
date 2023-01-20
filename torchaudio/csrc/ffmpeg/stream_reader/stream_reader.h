#pragma once
#include <torchaudio/csrc/ffmpeg/stream_reader/decoder.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_processor.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/typedefs.h>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

class StreamReader {
  AVFormatInputContextPtr pFormatContext;
  AVPacketPtr pPacket;

  std::vector<std::unique_ptr<StreamProcessor>> processors;
  // Mapping from user-facing stream index to internal index.
  // The first one is processor index,
  // the second is the map key inside of processor.
  std::vector<std::pair<int, int>> stream_indices;

  // timestamp to seek to expressed in AV_TIME_BASE
  //
  // 0 : No seek
  // Positive value: Skip AVFrames with timestamps before it
  // Negative value: UB. Should not happen
  //
  // Note:
  // When precise seek is performed, this value is set to the value provided
  // by client code, and PTS values of decoded frames are compared against it
  // to determine whether the frames should be passed to downstream.
  int64_t seek_timestamp = 0;

 public:
  explicit StreamReader(AVFormatInputContextPtr&& p);
  ~StreamReader() = default;
  // Non-copyable
  StreamReader(const StreamReader&) = delete;
  StreamReader& operator=(const StreamReader&) = delete;
  // Movable
  StreamReader(StreamReader&&) = default;
  StreamReader& operator=(StreamReader&&) = default;

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
  int64_t find_best_audio_stream() const;
  int64_t find_best_video_stream() const;
  // Fetch metadata of the source
  OptionDict get_metadata() const;
  // Fetch information about source streams
  int64_t num_src_streams() const;
  SrcStreamInfo get_src_stream_info(int i) const;
  // Fetch information about output streams
  int64_t num_out_streams() const;
  OutputStreamInfo get_out_stream_info(int i) const;
  // Check if all the buffers of the output streams are ready.
  bool is_buffer_ready() const;

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
  void seek(double timestamp_s, int64_t mode);

  void add_audio_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const c10::optional<OptionDict>& decoder_option);
  void add_video_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const c10::optional<OptionDict>& decoder_option,
      const c10::optional<std::string>& hw_accel);
  void remove_stream(int64_t i);

 private:
  void add_stream(
      int i,
      AVMediaType media_type,
      int frames_per_chunk,
      int num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const c10::optional<OptionDict>& decoder_option,
      const torch::Device& device);

 public:
  //////////////////////////////////////////////////////////////////////////////
  // Stream methods
  //////////////////////////////////////////////////////////////////////////////
  int process_packet();
  int process_packet_block(double timeout, double backoff);

 private:
  int drain();

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
 public:
  std::vector<c10::optional<torch::Tensor>> pop_chunks();
};

} // namespace ffmpeg
} // namespace torchaudio
