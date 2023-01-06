#include <torchaudio/csrc/ffmpeg/stream_reader/stream_reader_tensor_binding.h>

namespace torchaudio {
namespace ffmpeg {
namespace {

static int read_function(void* opaque, uint8_t* buf, int buf_size) {
  TensorIndexer* tensorobj = static_cast<TensorIndexer*>(opaque);

  int num_read = FFMIN(tensorobj->numel - tensorobj->index, buf_size);
  if (num_read == 0) {
    return AVERROR_EOF;
  }

  uint8_t* head = const_cast<uint8_t*>(tensorobj->data) + tensorobj->index;
  memcpy(buf, head, num_read);
  tensorobj->index += num_read;
  return num_read;
}

static int64_t seek_function(void* opaque, int64_t offset, int whence) {
  TensorIndexer* tensorobj = static_cast<TensorIndexer*>(opaque);

  if (whence == AVSEEK_SIZE) {
    return static_cast<int64_t>(tensorobj->numel);
  }

  if (whence == SEEK_SET) {
    tensorobj->index = offset;
  } else if (whence == SEEK_CUR) {
    tensorobj->index += offset;
  } else if (whence == SEEK_END) {
    tensorobj->index = tensorobj->numel + offset;
  } else {
    TORCH_CHECK(false, "[INTERNAL ERROR] Unexpected whence value: ", whence);
  }
  return static_cast<int64_t>(tensorobj->index);
}

AVIOContext* get_io_context(TensorIndexer* opaque, int buffer_size) {
  uint8_t* buffer = static_cast<uint8_t*>(av_malloc(buffer_size));
  TORCH_CHECK(buffer, "Failed to allocate buffer.");

  AVIOContext* av_io_ctx = avio_alloc_context(
      buffer,
      buffer_size,
      0,
      static_cast<void*>(opaque),
      &read_function,
      nullptr,
      &seek_function);
  if (!av_io_ctx) {
    av_freep(&buffer);
    TORCH_CHECK(av_io_ctx, "Failed to initialize AVIOContext.");
  }
  return av_io_ctx;
}

std::string get_id(const torch::Tensor& src) {
  std::stringstream ss;
  ss << "Tensor <" << static_cast<const void*>(src.data_ptr<uint8_t>()) << ">";
  return ss.str();
}
} // namespace

TensorIndexer::TensorIndexer(const torch::Tensor& src, int buffer_size)
    : src(src),
      data([&]() -> uint8_t* {
        TORCH_CHECK(
            src.is_contiguous(), "The input Tensor must be contiguous.");
        TORCH_CHECK(
            src.dtype() == torch::kUInt8,
            "The input Tensor must be uint8 type. Found: ",
            src.dtype());
        TORCH_CHECK(
            src.device().type() == c10::DeviceType::CPU,
            "The input Tensor must be on CPU. Found: ",
            src.device().str());
        TORCH_CHECK(
            src.dim() == 1, "The input Tensor must be 1D. Found: ", src.dim());
        return src.data_ptr<uint8_t>();
      }()),
      numel(src.numel()),
      pAVIO(get_io_context(this, buffer_size)) {}

StreamReaderTensorBinding::StreamReaderTensorBinding(
    const torch::Tensor& src,
    const c10::optional<std::string>& device,
    const c10::optional<OptionDict>& option,
    int buffer_size)
    : TensorIndexer(src, buffer_size),
      StreamReaderBinding(
          get_input_format_context(get_id(src), device, option, pAVIO)) {}

namespace {

c10::intrusive_ptr<StreamReaderTensorBinding> init(
    const torch::Tensor& src,
    const c10::optional<std::string>& device,
    const c10::optional<OptionDict>& option,
    int64_t buffer_size) {
  return c10::make_intrusive<StreamReaderTensorBinding>(
      src, device, option, static_cast<int>(buffer_size));
}

using S = const c10::intrusive_ptr<StreamReaderTensorBinding>&;

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.class_<StreamReaderTensorBinding>("ffmpeg_StreamReaderTensor")
      .def(torch::init<>(init))
      .def("num_src_streams", [](S self) { return self->num_src_streams(); })
      .def("num_out_streams", [](S self) { return self->num_out_streams(); })
      .def("get_metadata", [](S self) { return self->get_metadata(); })
      .def(
          "get_src_stream_info",
          [](S s, int64_t i) { return s->get_src_stream_info(i); })
      .def(
          "get_out_stream_info",
          [](S s, int64_t i) { return s->get_out_stream_info(i); })
      .def(
          "find_best_audio_stream",
          [](S s) { return s->find_best_audio_stream(); })
      .def(
          "find_best_video_stream",
          [](S s) { return s->find_best_video_stream(); })
      .def("seek", [](S s, double t, int64_t mode) { return s->seek(t, mode); })
      .def(
          "add_audio_stream",
          [](S s,
             int64_t i,
             int64_t frames_per_chunk,
             int64_t num_chunks,
             const c10::optional<std::string>& filter_desc,
             const c10::optional<std::string>& decoder,
             const c10::optional<OptionDict>& decoder_option) {
            s->add_audio_stream(
                i,
                frames_per_chunk,
                num_chunks,
                filter_desc,
                decoder,
                decoder_option);
          })
      .def(
          "add_video_stream",
          [](S s,
             int64_t i,
             int64_t frames_per_chunk,
             int64_t num_chunks,
             const c10::optional<std::string>& filter_desc,
             const c10::optional<std::string>& decoder,
             const c10::optional<OptionDict>& decoder_option,
             const c10::optional<std::string>& hw_accel) {
            s->add_video_stream(
                i,
                frames_per_chunk,
                num_chunks,
                filter_desc,
                decoder,
                decoder_option,
                hw_accel);
          })
      .def("remove_stream", [](S s, int64_t i) { s->remove_stream(i); })
      .def(
          "process_packet",
          [](S s, const c10::optional<double>& timeout, const double backoff) {
            return s->process_packet(timeout, backoff);
          })
      .def("process_all_packets", [](S s) { s->process_all_packets(); })
      .def(
          "fill_buffer",
          [](S s, const c10::optional<double>& timeout, const double backoff) {
            return s->fill_buffer(timeout, backoff);
          })
      .def("is_buffer_ready", [](S s) { return s->is_buffer_ready(); })
      .def("pop_chunks", [](S s, bool return_view) { return s->pop_chunks(return_view); });
}
} // namespace
} // namespace ffmpeg
} // namespace torchaudio
