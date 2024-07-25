#pragma once

#include <libtorio/ffmpeg/ffmpeg.h>
#include <libtorio/ffmpeg/filter_graph.h>
#include <libtorio/ffmpeg/stream_writer/encode_process.h>
#include <libtorio/ffmpeg/stream_writer/packet_writer.h>
#include <libtorio/ffmpeg/stream_writer/types.h>
#include <torch/types.h>

namespace torio {
namespace io {

////////////////////////////////////////////////////////////////////////////////
// StreamingMediaEncoder
////////////////////////////////////////////////////////////////////////////////

///
/// Encode and write audio/video streams chunk by chunk
///
class StreamingMediaEncoder {
  AVFormatOutputContextPtr format_ctx;
  std::map<int, EncodeProcess> processes;
  std::map<int, PacketWriter> packet_writers;

  AVPacketPtr pkt{alloc_avpacket()};
  bool is_open = false;
  int current_key = 0;

  /// @cond

 private:
  explicit StreamingMediaEncoder(AVFormatContext*);

 protected:
  /// Construct StreamingMediaEncoder from custom IO
  ///
  /// @param io_ctx Custom IO.
  /// @param format Specify output format.
  explicit StreamingMediaEncoder(
      AVIOContext* io_ctx,
      const std::optional<std::string>& format = c10::nullopt);

  /// @endcond

 public:
  /// Construct StreamingMediaEncoder from destination URI
  ///
  /// @param dst Destination where encoded data are written.
  /// @param format Specify output format. If not provided, it is guessed from
  /// ``dst``.
  explicit StreamingMediaEncoder(
      const std::string& dst,
      const std::optional<std::string>& format = c10::nullopt);

  // Non-copyable
  StreamingMediaEncoder(const StreamingMediaEncoder&) = delete;
  StreamingMediaEncoder& operator=(const StreamingMediaEncoder&) = delete;

  //////////////////////////////////////////////////////////////////////////////
  // Query methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// @cond

  /// Print the configured outputs
  void dump_format(int64_t i);

  /// @endcond

  //////////////////////////////////////////////////////////////////////////////
  // Configure methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// Add an output audio stream.
  ///
  /// @param sample_rate The sample rate.
  /// @param num_channels The number of channels.
  /// @param format Input sample format, which determines the dtype
  /// of the input tensor.
  /// @parblock
  ///
  /// - ``"u8"``: The input tensor must be ``torch.uint8`` type.
  /// - ``"s16"``: The input tensor must be ``torch.int16`` type.
  /// - ``"s32"``: The input tensor must be ``torch.int32`` type.
  /// - ``"s64"``: The input tensor must be ``torch.int64`` type.
  /// - ``"flt"``: The input tensor must be ``torch.float32`` type.
  /// - ``"dbl"``: The input tensor must be ``torch.float64`` type.
  ///
  /// Default: ``"flt"``.
  /// @endparblock
  /// @param encoder The name of the encoder to be used.
  /// @parblock
  /// When provided, use the specified encoder instead of the default one.
  ///
  /// To list the available encoders, you can use ``ffmpeg -encoders`` command.
  /// @endparblock
  /// @param encoder_option Options passed to encoder.
  /// To list encoder options for a encoder, you can use
  /// ``ffmpeg -h encoder=<ENCODER>``.
  /// @param encoder_format Format used to encode media.
  /// When encoder supports multiple formats, passing this argument will
  /// override the format used for encoding.
  ///  To list supported formats for the encoder, you can use
  /// ``ffmpeg -h encoder=<ENCODER>`` command.
  /// @param encoder_sample_rate If provided, perform resampling
  /// before encoding.
  /// @param encoder_num_channels If provided, change channel configuration
  /// before encoding.
  /// @param codec_config Codec configuration.
  /// @param filter_desc Additional processing to apply before
  /// encoding the input data
  void add_audio_stream(
      int sample_rate,
      int num_channels,
      const std::string& format,
      const std::optional<std::string>& encoder = c10::nullopt,
      const std::optional<OptionDict>& encoder_option = c10::nullopt,
      const std::optional<std::string>& encoder_format = c10::nullopt,
      const std::optional<int>& encoder_sample_rate = c10::nullopt,
      const std::optional<int>& encoder_num_channels = c10::nullopt,
      const std::optional<CodecConfig>& codec_config = c10::nullopt,
      const std::optional<std::string>& filter_desc = c10::nullopt);

  /// Add an output video stream.
  ///
  /// @param frame_rate Frame rate
  /// @param width Width
  /// @param height Height
  /// @param format Input pixel format, which determines the
  /// color channel order of the input tensor.
  /// @parblock
  ///
  /// - ``"gray8"``: One channel, grayscale.
  /// - ``"rgb24"``: Three channels in the order of RGB.
  /// - ``"bgr24"``: Three channels in the order of BGR.
  /// - ``"yuv444p"``: Three channels in the order of YUV.
  ///
  /// In either case, the input tensor has to be ``torch.uint8`` type and
  /// the shape must be (frame, channel, height, width).
  /// @endparblock
  /// @param encoder See ``add_audio_stream()``.
  /// @param encoder_option See ``add_audio_stream()``.
  /// @param encoder_format See ``add_audio_stream()``.
  /// @param encoder_frame_rate If provided, change frame rate before encoding.
  /// @param encoder_width If provided, resize image before encoding.
  /// @param encoder_height If provided, resize image before encoding.
  /// @param hw_accel Enable hardware acceleration.
  /// @param codec_config Codec configuration.
  /// @parblock
  /// When video is encoded on CUDA hardware, for example
  /// `encoder="h264_nvenc"`, passing CUDA device indicator to `hw_accel`
  /// (i.e. `hw_accel="cuda:0"`) will make StreamingMediaEncoder expect video
  /// chunk to be a CUDA Tensor. Passing CPU Tensor will result in an error.
  ///
  /// If `None`, the video chunk Tensor has to be a CPU Tensor.
  /// @endparblock
  /// @param filter_desc Additional processing to apply before
  /// encoding the input data
  void add_video_stream(
      double frame_rate,
      int width,
      int height,
      const std::string& format,
      const std::optional<std::string>& encoder = c10::nullopt,
      const std::optional<OptionDict>& encoder_option = c10::nullopt,
      const std::optional<std::string>& encoder_format = c10::nullopt,
      const std::optional<double>& encoder_frame_rate = c10::nullopt,
      const std::optional<int>& encoder_width = c10::nullopt,
      const std::optional<int>& encoder_height = c10::nullopt,
      const std::optional<std::string>& hw_accel = c10::nullopt,
      const std::optional<CodecConfig>& codec_config = c10::nullopt,
      const std::optional<std::string>& filter_desc = c10::nullopt);
  /// @cond
  /// Add output audio frame stream.
  /// Allows for writing frames rather than tensors via `write_frame`.
  ///
  /// See `add_audio_stream` for more detail on input parameters.
  void add_audio_frame_stream(
      int sample_rate,
      int num_channels,
      const std::string& format,
      const std::optional<std::string>& encoder = c10::nullopt,
      const std::optional<OptionDict>& encoder_option = c10::nullopt,
      const std::optional<std::string>& encoder_format = c10::nullopt,
      const std::optional<int>& encoder_sample_rate = c10::nullopt,
      const std::optional<int>& encoder_num_channels = c10::nullopt,
      const std::optional<CodecConfig>& codec_config = c10::nullopt,
      const std::optional<std::string>& filter_desc = c10::nullopt);

  /// Add output video frame stream.
  /// Allows for writing frames rather than tensors via `write_frame`.
  ///
  /// See `add_video_stream` for more detail on input parameters.
  void add_video_frame_stream(
      double frame_rate,
      int width,
      int height,
      const std::string& format,
      const std::optional<std::string>& encoder = c10::nullopt,
      const std::optional<OptionDict>& encoder_option = c10::nullopt,
      const std::optional<std::string>& encoder_format = c10::nullopt,
      const std::optional<double>& encoder_frame_rate = c10::nullopt,
      const std::optional<int>& encoder_width = c10::nullopt,
      const std::optional<int>& encoder_height = c10::nullopt,
      const std::optional<std::string>& hw_accel = c10::nullopt,
      const std::optional<CodecConfig>& codec_config = c10::nullopt,
      const std::optional<std::string>& filter_desc = c10::nullopt);

  /// Add packet stream. Intended to be used in conjunction with
  /// ``StreamingMediaDecoder`` to perform packet passthrough.
  /// @param stream_params Stream parameters returned by
  /// ``StreamingMediaDecoder::get_src_stream_params()`` for the packet stream
  /// to pass through.
  void add_packet_stream(const StreamParams& stream_params);

  /// @endcond

  /// Set file-level metadata
  /// @param metadata metadata.
  void set_metadata(const OptionDict& metadata);

  //////////////////////////////////////////////////////////////////////////////
  // Write methods
  //////////////////////////////////////////////////////////////////////////////
 public:
  /// Open the output file / device and write the header.
  ///
  /// @param opt Private options for protocol, device and muxer.
  void open(const std::optional<OptionDict>& opt = c10::nullopt);
  /// Close the output file / device and finalize metadata.
  void close();

  /// Write audio data
  /// @param i Stream index.
  /// @param frames Waveform tensor. Shape: ``(frame, channel)``.
  /// The ``dtype`` must match what was passed to ``add_audio_stream()`` method.
  /// @param pts
  /// @parblock
  /// Presentation timestamp. If provided, it overwrites the PTS of
  /// the first frame with the provided one. Otherwise, PTS are incremented per
  /// an inverse of sample rate. Only values exceed the PTS values processed
  /// internally.
  ///
  /// __NOTE__: The provided value is converted to integer value expressed
  /// in basis of sample rate.
  /// Therefore, it is truncated to the nearest value of ``n / sample_rate``.
  /// @endparblock
  void write_audio_chunk(
      int i,
      const torch::Tensor& frames,
      const std::optional<double>& pts = c10::nullopt);
  /// Write video data
  /// @param i Stream index.
  /// @param frames Video/image tensor. Shape: ``(time, channel, height,
  /// width)``. The ``dtype`` must be ``torch.uint8``. The shape ``(height,
  /// width and the number of channels)`` must match what was configured when
  /// calling ``add_video_stream()``.
  /// @param pts
  /// @parblock
  /// Presentation timestamp. If provided, it overwrites the PTS of
  /// the first frame with the provided one. Otherwise, PTS are incremented per
  /// an inverse of frame rate. Only values exceed the PTS values processed
  /// internally.
  ///
  /// __NOTE__: The provided value is converted to integer value expressed
  /// in basis of frame rate.
  /// Therefore, it is truncated to the nearest value of ``n / frame_rate``.
  /// @endparblock
  void write_video_chunk(
      int i,
      const torch::Tensor& frames,
      const std::optional<double>& pts = c10::nullopt);
  /// @cond
  /// Write frame to stream.
  /// @param i Stream index.
  /// @param frame Frame to write.
  void write_frame(int i, AVFrame* frame);
  /// Write packet.
  /// @param packet Packet to write, passed from ``StreamingMediaDecoder``.
  void write_packet(const AVPacketPtr& packet);
  /// @endcond

  /// Flush the frames from encoders and write the frames to the destination.
  void flush();

 private:
  int num_output_streams();
};

////////////////////////////////////////////////////////////////////////////////
// StreamingMediaEncoderCustomIO
////////////////////////////////////////////////////////////////////////////////

/// @cond

namespace detail {
struct CustomOutput {
  AVIOContextPtr io_ctx;
  CustomOutput(
      void* opaque,
      int buffer_size,
      int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence));
};
} // namespace detail

/// @endcond

///
/// A subclass of StreamingMediaDecoder which works with custom read function.
/// Can be used for encoding media into memory or custom object.
///
class StreamingMediaEncoderCustomIO : private detail::CustomOutput,
                                      public StreamingMediaEncoder {
 public:
  /// Construct StreamingMediaEncoderCustomIO with custom write and seek
  /// functions.
  ///
  /// @param opaque Custom data used by ``write_packet`` and ``seek`` functions.
  /// @param format Specify output format.
  /// @param buffer_size The size of the intermediate buffer, which FFmpeg uses
  /// to pass data to write_packet function.
  /// @param write_packet Custom write function that is called from FFmpeg to
  /// actually write data to the custom destination.
  /// @param seek Optional seek function that is used to seek the destination.
  StreamingMediaEncoderCustomIO(
      void* opaque,
      const std::optional<std::string>& format,
      int buffer_size,
      int (*write_packet)(void* opaque, uint8_t* buf, int buf_size),
      int64_t (*seek)(void* opaque, int64_t offset, int whence) = nullptr);
};

// For BC
using StreamWriter = StreamingMediaEncoder;
using StreamWriterCustomIO = StreamingMediaEncoderCustomIO;

} // namespace io
} // namespace torio

// For BC
namespace torchaudio::io {
using namespace torio::io;
} // namespace torchaudio::io
