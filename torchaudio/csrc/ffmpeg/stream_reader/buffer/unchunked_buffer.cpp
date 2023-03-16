#include <torchaudio/csrc/ffmpeg/stream_reader/buffer/unchunked_buffer.h>
#include <torchaudio/csrc/ffmpeg/stream_reader/conversion.h>

namespace torchaudio {
namespace io {
namespace detail {

template <typename Converter>
UnchunkedBuffer<Converter>::UnchunkedBuffer(Converter&& converter_)
    : converter(std::move(converter_)) {}

template <typename Converter>
bool UnchunkedBuffer<Converter>::is_ready() const {
  return chunks.size() > 0;
}

template <typename Converter>
void UnchunkedBuffer<Converter>::push_frame(AVFrame* frame, double pts_) {
  if (chunks.size() == 0) {
    pts = pts_;
  }
  chunks.push_back(converter.convert(frame));
}

template <typename Converter>
c10::optional<Chunk> UnchunkedBuffer<Converter>::pop_chunk() {
  if (chunks.size() == 0) {
    return {};
  }

  auto frames =
      torch::cat(std::vector<torch::Tensor>{chunks.begin(), chunks.end()}, 0);
  chunks.clear();
  return {Chunk{frames, pts}};
}

template <typename Converter>
void UnchunkedBuffer<Converter>::flush() {
  chunks.clear();
}

std::unique_ptr<Buffer> get_unchunked_buffer(AVSampleFormat fmt, int channels) {
  switch (fmt) {
    case AV_SAMPLE_FMT_U8: {
      using Converter = AudioConverter<torch::kUInt8, false>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_S16: {
      using Converter = AudioConverter<torch::kInt16, false>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_S32: {
      using Converter = AudioConverter<torch::kInt32, false>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_S64: {
      using Converter = AudioConverter<torch::kInt64, false>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_FLT: {
      using Converter = AudioConverter<torch::kFloat32, false>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_DBL: {
      using Converter = AudioConverter<torch::kFloat64, false>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_U8P: {
      using Converter = AudioConverter<torch::kUInt8, true>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_S16P: {
      using Converter = AudioConverter<torch::kInt16, true>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_S32P: {
      using Converter = AudioConverter<torch::kInt32, true>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_S64P: {
      using Converter = AudioConverter<torch::kInt64, true>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_FLTP: {
      using Converter = AudioConverter<torch::kFloat32, true>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    case AV_SAMPLE_FMT_DBLP: {
      using Converter = AudioConverter<torch::kFloat64, true>;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{channels});
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unexpected audio type:", av_get_sample_fmt_name(fmt));
  }
}

std::unique_ptr<Buffer> get_unchunked_buffer(
    AVPixelFormat fmt,
    int h,
    int w,
    const torch::Device& device) {
  if (device.type() == at::DeviceType::CUDA) {
#ifndef USE_CUDA
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        false,
        "USE_CUDA is not defined, and it should be guarded before here.");
#else
    switch (fmt) {
      case AV_PIX_FMT_NV12: {
        using Conv = NV12CudaConverter;
        return std::make_unique<UnchunkedBuffer<Conv>>(Conv{h, w, device});
      }
      case AV_PIX_FMT_P010: {
        using Conv = P010CudaConverter;
        return std::make_unique<UnchunkedBuffer<Conv>>(Conv{h, w, device});
      }
      case AV_PIX_FMT_P016: {
        TORCH_CHECK(
            false,
            "Unsupported video format found in CUDA HW: ",
            av_get_pix_fmt_name(fmt));
      }
      default: {
        TORCH_CHECK(
            false,
            "Unexpected video format found in CUDA HW: ",
            av_get_pix_fmt_name(fmt));
      }
    }
#endif
  }

  switch (fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24: {
      using Converter = InterlacedImageConverter;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{h, w, 3});
    }
    case AV_PIX_FMT_ARGB:
    case AV_PIX_FMT_RGBA:
    case AV_PIX_FMT_ABGR:
    case AV_PIX_FMT_BGRA: {
      using Converter = InterlacedImageConverter;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{h, w, 4});
    }
    case AV_PIX_FMT_GRAY8: {
      using Converter = InterlacedImageConverter;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{h, w, 1});
    }
    case AV_PIX_FMT_RGB48LE: {
      using Converter = Interlaced16BitImageConverter;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{h, w, 3});
    }
    case AV_PIX_FMT_YUV444P: {
      using Converter = PlanarImageConverter;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{h, w, 3});
    }
    case AV_PIX_FMT_YUV420P: {
      using Converter = YUV420PConverter;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{h, w});
    }
    case AV_PIX_FMT_NV12: {
      using Converter = NV12Converter;
      return std::make_unique<UnchunkedBuffer<Converter>>(Converter{h, w});
    }
    default: {
      TORCH_INTERNAL_ASSERT(
          false, "Unexpected video format found: ", av_get_pix_fmt_name(fmt));
    }
  }
}

} // namespace detail
} // namespace io
} // namespace torchaudio
