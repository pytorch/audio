#pragma once

#include <torch/torch.h>
#include <torchaudio/csrc/ffmpeg/ffmpeg.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/decoder.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/sink.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/typedefs.h>
#include <map>

namespace torchaudio {
namespace io {

class StreamProcessor {
 public:
  using KeyType = int;

 private:
  // Link to the corresponding stream object
  const AVStream* stream;

  // Components for decoding source media
  AVFramePtr pFrame1;
  AVFramePtr pFrame2;
  Decoder decoder;

  KeyType current_key = 0;
  std::map<KeyType, Sink> sinks;

  // Used for precise seek.
  // 0: no discard
  // Positive Values: decoded frames with PTS values less than this are
  // discarded.
  // Negative values: UB. Should not happen.
  int64_t discard_before_pts = 0;

 public:
  StreamProcessor(
      AVStream* stream,
      const c10::optional<std::string>& decoder_name,
      const c10::optional<OptionDict>& decoder_option,
      const torch::Device& device);
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
      const c10::optional<std::string>& filter_description,
      const torch::Device& device);

  // 1. Remove the stream
  void remove_stream(KeyType key);

  // Set discard
  // The input timestamp must be expressed in AV_TIME_BASE unit.
  void set_discard_timestamp(int64_t timestamp);

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
  std::string get_filter_description(KeyType key) const;
  bool is_buffer_ready() const;

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
  c10::optional<Chunk> pop_chunk(KeyType key);
};

} // namespace io
} // namespace torchaudio
