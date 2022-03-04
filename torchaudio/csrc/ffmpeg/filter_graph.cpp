#include <torchaudio/csrc/ffmpeg/filter_graph.h>
#include <stdexcept>

namespace torchaudio {
namespace ffmpeg {

FilterGraph::FilterGraph(
    AVRational time_base,
    AVCodecParameters* codecpar,
    std::string filter_description)
    : input_time_base(time_base),
      codecpar(codecpar),
      filter_description(std::move(filter_description)),
      media_type(codecpar->codec_type) {
  init();
}

////////////////////////////////////////////////////////////////////////////////
// Query method
////////////////////////////////////////////////////////////////////////////////
std::string FilterGraph::get_description() const {
  return filter_description;
};

////////////////////////////////////////////////////////////////////////////////
// Configuration methods
////////////////////////////////////////////////////////////////////////////////
namespace {
std::string get_audio_src_args(
    AVRational time_base,
    AVCodecParameters* codecpar) {
  char args[512];
  std::snprintf(
      args,
      sizeof(args),
      "time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=0x%" PRIx64,
      time_base.num,
      time_base.den,
      codecpar->sample_rate,
      av_get_sample_fmt_name(static_cast<AVSampleFormat>(codecpar->format)),
      codecpar->channel_layout);
  return std::string(args);
}

std::string get_video_src_args(
    AVRational time_base,
    AVCodecParameters* codecpar) {
  char args[512];
  std::snprintf(
      args,
      sizeof(args),
      "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
      codecpar->width,
      codecpar->height,
      static_cast<AVPixelFormat>(codecpar->format),
      time_base.num,
      time_base.den,
      codecpar->sample_aspect_ratio.num,
      codecpar->sample_aspect_ratio.den);
  return std::string(args);
}

} // namespace

void FilterGraph::init() {
  add_src();
  add_sink();
  add_process();
  create_filter();
}

void FilterGraph::reset() {
  pFilterGraph.reset();
  buffersrc_ctx = nullptr;
  buffersink_ctx = nullptr;

  init();
}

void FilterGraph::add_src() {
  std::string args;
  switch (media_type) {
    case AVMEDIA_TYPE_AUDIO:
      args = get_audio_src_args(input_time_base, codecpar);
      break;
    case AVMEDIA_TYPE_VIDEO:
      args = get_video_src_args(input_time_base, codecpar);
      break;
    default:
      throw std::runtime_error("Only audio/video are supported.");
  }

  const AVFilter* buffersrc = avfilter_get_by_name(
      media_type == AVMEDIA_TYPE_AUDIO ? "abuffer" : "buffer");
  int ret = avfilter_graph_create_filter(
      &buffersrc_ctx, buffersrc, "in", args.c_str(), NULL, pFilterGraph);
  if (ret < 0) {
    throw std::runtime_error(
        "Failed to create input filter: \"" + args + "\" (" +
        av_err2string(ret) + ")");
  }
}

void FilterGraph::add_sink() {
  if (media_type == AVMEDIA_TYPE_UNKNOWN) {
    throw std::runtime_error("Source buffer is not allocated.");
  }
  if (buffersink_ctx) {
    throw std::runtime_error("Sink buffer is already allocated.");
  }
  const AVFilter* buffersink = avfilter_get_by_name(
      media_type == AVMEDIA_TYPE_AUDIO ? "abuffersink" : "buffersink");
  // Note
  // Originally, the code here followed the example
  // https://ffmpeg.org/doxygen/4.1/filtering_audio_8c-example.html
  // which sets option for `abuffersink`, which caused an issue where the
  // `abuffersink` parameters set for the first time survive across multiple
  // fitler generations.
  // According to the other example
  // https://ffmpeg.org/doxygen/4.1/filter_audio_8c-example.html
  // `abuffersink` should not take options, and this resolved issue.
  int ret = avfilter_graph_create_filter(
      &buffersink_ctx, buffersink, "out", nullptr, nullptr, pFilterGraph);
  if (ret < 0) {
    throw std::runtime_error("Failed to create output filter.");
  }
}

namespace {

// Encapsulating AVFilterInOut* with handy methods since
// we need to deal with multiple of them at the same time.
class InOuts {
  AVFilterInOut* p = nullptr;
  // Disable copy constructor/assignment just in case.
  InOuts(const InOuts&) = delete;
  InOuts& operator=(const InOuts&) = delete;

 public:
  InOuts(const char* name, AVFilterContext* pCtx) {
    p = avfilter_inout_alloc();
    if (!p) {
      throw std::runtime_error("Failed to allocate AVFilterInOut.");
    }
    p->name = av_strdup(name);
    p->filter_ctx = pCtx;
    p->pad_idx = 0;
    p->next = nullptr;
  }
  ~InOuts() {
    avfilter_inout_free(&p);
  }
  operator AVFilterInOut**() {
    return &p;
  }
};

} // namespace

void FilterGraph::add_process() {
  // Note
  // The official example and other derived codes out there use
  // https://ffmpeg.org/doxygen/4.1/filtering_audio_8c-example.html#_a37
  // variable name `in` for "out"/buffersink, and `out` for "in"/buffersrc.
  // If you are debugging this part of the code, you might get confused.
  InOuts in{"in", buffersrc_ctx}, out{"out", buffersink_ctx};

  std::string desc = filter_description.empty()
      ? (media_type == AVMEDIA_TYPE_AUDIO) ? "anull" : "null"
      : filter_description;

  int ret =
      avfilter_graph_parse_ptr(pFilterGraph, desc.c_str(), out, in, nullptr);

  if (ret < 0) {
    throw std::runtime_error(
        "Failed to create the filter from \"" + desc + "\" (" +
        av_err2string(ret) + ".)");
  }
}

void FilterGraph::create_filter() {
  if (avfilter_graph_config(pFilterGraph, nullptr) < 0)
    throw std::runtime_error("Failed to configure the graph.");
  // char* desc = avfilter_graph_dump(pFilterGraph.get(), NULL);
  // std::cerr << "Filter created:\n" << desc << std::endl;
  // av_free(static_cast<void*>(desc));
}

////////////////////////////////////////////////////////////////////////////////
// Streaming process
//////////////////////////////////////////////////////////////////////////////
int FilterGraph::add_frame(AVFrame* pInputFrame) {
  return av_buffersrc_add_frame_flags(
      buffersrc_ctx, pInputFrame, AV_BUFFERSRC_FLAG_KEEP_REF);
}

int FilterGraph::get_frame(AVFrame* pOutputFrame) {
  return av_buffersink_get_frame(buffersink_ctx, pOutputFrame);
}

} // namespace ffmpeg
} // namespace torchaudio
