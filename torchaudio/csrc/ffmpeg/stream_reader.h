#pragma once
#include <torchaudio/csrc/export.h>
#include <torchaudio/csrc/ffmpeg/decoder.h>
#include <torchaudio/csrc/ffmpeg/stream_processor.h>
#include <torchaudio/csrc/ffmpeg/typedefs.h>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

class StreamReader {
  AVFormatContextPtr pFormatContext;
  AVPacketPtr pPacket;

  std::vector<std::unique_ptr<StreamProcessor>> processors;
  // Mapping from user-facing stream index to internal index.
  // The first one is processor index,
  // the second is the map key inside of processor.
  std::vector<std::pair<int, int>> stream_indices;

 public:
  TORCHAUDIO_API explicit StreamReader(AVFormatContextPtr&& p);
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
  TORCHAUDIO_API int64_t find_best_audio_stream() const;
  TORCHAUDIO_API int64_t find_best_video_stream() const;
  // Fetch metadata of the source
  TORCHAUDIO_API c10::Dict<std::string, std::string> get_metadata() const;
  // Fetch information about source streams
  TORCHAUDIO_API int64_t num_src_streams() const;
  TORCHAUDIO_API SrcStreamInfo get_src_stream_info(int i) const;
  // Fetch information about output streams
  TORCHAUDIO_API int64_t num_out_streams() const;
  TORCHAUDIO_API OutputStreamInfo get_out_stream_info(int i) const;
  // Check if all the buffers of the output streams are ready.
  TORCHAUDIO_API bool is_buffer_ready() const;

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
  TORCHAUDIO_API void seek(double timestamp);

  TORCHAUDIO_API void add_audio_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const OptionDict& decoder_option);
  TORCHAUDIO_API void add_video_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const OptionDict& decoder_option,
      const c10::optional<std::string>& hw_accel);
  TORCHAUDIO_API void remove_stream(int64_t i);

 private:
  void add_stream(
      int i,
      AVMediaType media_type,
      int frames_per_chunk,
      int num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const OptionDict& decoder_option,
      const torch::Device& device);

 public:
  //////////////////////////////////////////////////////////////////////////////
  // Stream methods
  //////////////////////////////////////////////////////////////////////////////
  TORCHAUDIO_API int process_packet();
  TORCHAUDIO_API int process_packet_block(double timeout, double backoff);

  TORCHAUDIO_API int drain();

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
  TORCHAUDIO_API std::vector<c10::optional<torch::Tensor>> pop_chunks();
};

} // namespace ffmpeg
} // namespace torchaudio
