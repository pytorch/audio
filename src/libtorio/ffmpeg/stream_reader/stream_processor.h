#pragma once

#include <libtorio/ffmpeg/ffmpeg.h>
#include <libtorio/ffmpeg/stream_reader/post_process.h>
#include <libtorio/ffmpeg/stream_reader/typedefs.h>
#include <torch/types.h>
#include <map>

namespace torio {
namespace io {

class StreamProcessor {
 public:
  using KeyType = int;

 private:
  // Stream time base which is not stored in AVCodecContextPtr
  AVRational stream_time_base;

  // Components for decoding source media
  AVCodecContextPtr codec_ctx{nullptr};
  AVFramePtr frame{alloc_avframe()};

  KeyType current_key = 0;
  std::map<KeyType, std::unique_ptr<IPostDecodeProcess>> post_processes;

  // Used for precise seek.
  // 0: no discard
  // Positive Values: decoded frames with PTS values less than this are
  // discarded.
  // Negative values: UB. Should not happen.
  int64_t discard_before_pts = 0;

 public:
  explicit StreamProcessor(const AVRational& time_base);
  ~StreamProcessor() = default;
  // Non-copyable
  StreamProcessor(const StreamProcessor&) = delete;
  StreamProcessor& operator=(const StreamProcessor&) = delete;
  // Movable
  StreamProcessor(StreamProcessor&&) = default;
  StreamProcessor& operator=(StreamProcessor&&) = default;

  //////////////////////////////////////////////////////////////////////////////
  // Configurations
  //////////////////////////////////////////////////////////////////////////////
  // 1. Initialize decoder (if not initialized yet)
  // 2. Configure a new audio/video filter.
  //    If the custom parameter is provided, then perform resize, resample etc..
  //    otherwise, the filter only converts the sample type.
  // 3. Configure a buffer.
  // 4. Return filter ID.
  KeyType add_stream(
      int frames_per_chunk,
      int num_chunks,
      AVRational frame_rate,
      const std::string& filter_description,
      const torch::Device& device);

  // 1. Remove the stream
  void remove_stream(KeyType key);

  // Set discard
  // The input timestamp must be expressed in AV_TIME_BASE unit.
  void set_discard_timestamp(int64_t timestamp);

  void set_decoder(
      const AVCodecParameters* codecpar,
      const std::optional<std::string>& decoder_name,
      const std::optional<OptionDict>& decoder_option,
      const torch::Device& device);

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
  [[nodiscard]] std::string get_filter_description(KeyType key) const;
  [[nodiscard]] FilterGraphOutputInfo get_filter_output_info(KeyType key) const;

  bool is_buffer_ready() const;
  [[nodiscard]] bool is_decoder_set() const;

  //////////////////////////////////////////////////////////////////////////////
  // The streaming process
  //////////////////////////////////////////////////////////////////////////////
  // 1. decode the input frame
  // 2. pass the decoded data to filters
  // 3. each filter store the result to the corresponding buffer
  // - Sending NULL will drain (flush) the internal
  int process_packet(AVPacket* packet);

  // flush the internal buffer of decoder.
  // To be use when seeking
  void flush();

 private:
  int send_frame(AVFrame* pFrame);

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
 public:
  // Get the chunk from the given filter result
  std::optional<Chunk> pop_chunk(KeyType key);
};

} // namespace io
} // namespace torio
