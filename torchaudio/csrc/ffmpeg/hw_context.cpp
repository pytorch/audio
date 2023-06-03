#include <torchaudio/csrc/ffmpeg/hw_context.h>
#include <torchaudio/csrc/ffmpeg/stub.h>

namespace torchaudio::io {
namespace {

static std::mutex MUTEX;
static std::map<int, AVBufferRefPtr> CUDA_CONTEXT_CACHE;

} // namespace

AVBufferRef* get_cuda_context(int index) {
  std::lock_guard<std::mutex> lock(MUTEX);
  if (index == -1) {
    index = 0;
  }
  if (CUDA_CONTEXT_CACHE.count(index) == 0) {
    AVBufferRef* p = nullptr;
    int ret = FFMPEG av_hwdevice_ctx_create(
        &p, AV_HWDEVICE_TYPE_CUDA, std::to_string(index).c_str(), nullptr, 0);
    TORCH_CHECK(
        ret >= 0,
        "Failed to create CUDA device context on device ",
        index,
        "(",
        av_err2string(ret),
        ")");
    assert(p);
    CUDA_CONTEXT_CACHE.emplace(index, p);
    return p;
  }
  AVBufferRefPtr& buffer = CUDA_CONTEXT_CACHE.at(index);
  return buffer;
}

void clear_cuda_context_cache() {
  std::lock_guard<std::mutex> lock(MUTEX);
  CUDA_CONTEXT_CACHE.clear();
}

} // namespace torchaudio::io
