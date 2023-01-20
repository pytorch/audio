#pragma once
#include <torchaudio/csrc/ffmpeg/stream_reader/decoder.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/stream_processor.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/typedefs.h>
#include <vector>

namespace torchaudio {
namespace ffmpeg {

///
/// Fetch and decode audio/video streams chunk by chunk.
///
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
  /// @name Constructors
  ///
  ///@{

  /// @todo Introduce a constructor that takes std::string and abstracts away
  /// ffmpeg-native structs
  ///
  explicit StreamReader(AVFormatInputContextPtr&& p);

  ///@}

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
  /// @name Query methods
  ///@{

  /// Find a suitable audio stream using heuristics from ffmpeg.
  ///
  /// If successful, the index of the best stream (>=0) is returned.
  /// Otherwise a negative value is returned.
  int64_t find_best_audio_stream() const;
  /// Find a suitable video stream using heuristics from ffmpeg.
  ///
  /// If successful, the index of the best stream (0>=) is returned.
  /// otherwise a negative value is returned.
  int64_t find_best_video_stream() const;
  /// Fetch metadata of the source media.
  OptionDict get_metadata() const;
  /// Fetch the number of source streams found in the input media.
  ///
  /// The source streams include not only audio/video streams but also
  /// subtitle and others.
  int64_t num_src_streams() const;
  /// Fetch information about the specified source stream.
  ///
  /// The valid value range is ``[0, num_src_streams())``.
  SrcStreamInfo get_src_stream_info(int i) const;
  /// Fetch the number of output streams defined by client code.
  ///
  /// @addtogroup stream_reader_query_methods
  int64_t num_out_streams() const;
  /// Fetch information about the specified output stream.
  ///
  /// The valid value range is ``[0, num_out_streams())``.
  OutputStreamInfo get_out_stream_info(int i) const;
  /// Check if all the buffers of the output streams have enough decoded frames.
  bool is_buffer_ready() const;

  ///@}

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
  /// @name Configure methods
  ///@{

  /// Define an output audio stream.
  ///
  /// @param i The index of the source stream.
  ///
  /// @param frames_per_chunk Number of frames returned as one chunk.
  /// @parblock
  ///   If a source stream is exhausted before ``frames_per_chunk``  frames
  ///   are buffered, the chunk is returned as-is. Thus the number of frames
  ///   in the chunk may be smaller than ````frames_per_chunk``.
  ///
  ///   Providing ``-1`` disables chunking, in which case, method
  /// ``pop_chunks()`` returns all the buffered frames as one chunk.
  /// @endparblock
  ///
  /// @param num_chunks Internal buffer size.
  /// @parblock
  ///   When the number of buffered chunks exceeds this number, old chunks are
  ///   dropped. For example, if `frames_per_chunk` is 5 and `buffer_chunk_size`
  ///   is 3, then frames older than 15 are dropped.
  ///
  ///   Providing ``-1`` disables this behavior, forcing the retention of all
  ///   chunks.
  /// @endparblock
  ///
  /// @param filter_desc Description of filter graph applied to the source
  /// stream.
  ///
  /// @param decoder The name of the decoder to be used.
  ///   When provided, use the specified decoder instead of the default one.
  ///
  /// @param decoder_option Options passed to decoder.
  /// @parblock
  ///   To list decoder options for a decoder, you can use
  ///   `ffmpeg -h decoder=<DECODER>` command.
  ///
  ///   In addition to decoder-specific options, you can also pass options
  ///   related to multithreading. They are effective only if the decoder
  ///   supports them. If neither of them are provided, StreamReader defaults to
  ///   single thread.
  ///    - ``"threads"``: The number of threads or the value ``"0"``
  ///      to let FFmpeg decide based on its heuristics.
  ///    - ``"thread_type"``: Which multithreading method to use.
  ///      The valid values are ``"frame"`` or ``"slice"``.
  ///      Note that each decoder supports a different set of methods.
  ///      If not provided, a default value is used.
  ///       - ``"frame"``: Decode more than one frame at once.
  ///         Each thread handles one frame.
  ///         This will increase decoding delay by one frame per thread
  ///       - ``"slice"``: Decode more than one part of a single frame at once.
  /// @endparblock
  void add_audio_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const c10::optional<OptionDict>& decoder_option);
  /// Define an output video stream.
  ///
  /// @param i,frames_per_chunk,num_chunks,filter_desc,decoder,decoder_option
  /// See `add_audio_stream()`.
  ///
  /// @param hw_accel Enable hardware acceleration.
  /// @parblock
  /// When video is decoded on CUDA hardware, (for example by specifying
  /// `"h264_cuvid"` decoder), passing CUDA device indicator to ``hw_accel``
  /// (i.e. ``hw_accel="cuda:0"``) will make StreamReader place the resulting
  /// frames directly on the specified CUDA device as a CUDA tensor.
  ///
  /// If `None`, the chunk will be moved to CPU memory.
  /// @endparblock
  void add_video_stream(
      int64_t i,
      int64_t frames_per_chunk,
      int64_t num_chunks,
      const c10::optional<std::string>& filter_desc,
      const c10::optional<std::string>& decoder,
      const c10::optional<OptionDict>& decoder_option,
      const c10::optional<std::string>& hw_accel);
  /// Remove an output stream.
  ///
  /// @param i The index of the output stream to be removed.
  /// The valid value range is `[0, num_out_streams())`.
  void remove_stream(int64_t i);

  ///@}

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

  //////////////////////////////////////////////////////////////////////////////
  // Stream methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// @name Stream methods
  ///@{

  /// Seek into the given time stamp.
  ///
  /// @param timestamp Target time stamp in second.
  /// @param mode Seek mode.
  /// - ``0``: Keyframe mode. Seek into nearest key frame before the given
  /// timestamp.
  /// - ``1``: Any mode. Seek into any frame (including non-key frames) before
  ///   the given timestamp.
  /// - ``2``: Precise mode. First seek into the nearest key frame before the
  ///   given timestamp, then decode frames until it reaches the frame closest
  ///   to the given timestamp.
  void seek(double timestamp, int64_t mode);

  /// Demultiplex and process one packet.
  ///
  /// @return
  /// - ``0``: A packet was processed successfully and there are still
  ///   packets left in the stream, so client code can call this method again.
  /// - ``1``: A packet was processed successfully and it reached EOF.
  ///   Client code should not call this method again.
  /// - ``<0``: An error has happened.
  int process_packet();
  /// Similar to `process_packet()`, but in case it fails due to resource
  /// temporarily being unavailable, it automatically retries.
  ///
  /// This behavior is helpful when using device input, such as a microphone,
  /// during which the buffer may be busy while sample acquisition is happening.
  ///
  /// @param timeout Timeout in milli seconds.
  /// - ``>=0``: Keep retrying until the given time passes.
  /// - ``<0``: Keep retrying forever.
  /// @param backoff Time to wait before retrying in milli seconds.
  int process_packet_block(double timeout, double backoff);

  ///@}

 private:
  int drain();

  //////////////////////////////////////////////////////////////////////////////
  // Retrieval
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// @name Retrieval methods
  ///@{

  /// Pop one chunk from each output stream if it is available.
  std::vector<c10::optional<torch::Tensor>> pop_chunks();

  ///@}
};

} // namespace ffmpeg
} // namespace torchaudio
