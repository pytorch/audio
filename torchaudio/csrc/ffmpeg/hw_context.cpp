#include <torchaudio/csrc/ffmpeg/hw_context.h>
#include <cuda_runtime.h>
#include <ATen/DynamicLibrary.h>
#include <libavutil/hwcontext_cuda.h>

namespace torchaudio::io {
namespace {

static std::mutex MUTEX;
static std::map<int, AVBufferRefPtr> CUDA_CONTEXT_CACHE;

} // namespace

CUresult (*cuDeviceGet_) (CUdevice*, int) = nullptr;
CUresult (*cuDevicePrimaryCtxGetState_) (CUdevice, unsigned int*, int*) = nullptr;

  #define CUDA_CHECK(X)                                   \
  do {                                                  \
    auto result = X;                                    \
    if (result != CUDA_SUCCESS) {                        \
      fprintf(                                          \
          stderr,                                       \
          "File %s Line %d %s returned %d.\n",          \
          __FILE__,                                     \
          __LINE__,                                     \
          #X,X);                                        \
      abort();                                          \
    }                                                   \
  } while (0)

AVBufferRef* get_cuda_context(int index) {
  std::lock_guard<std::mutex> lock(MUTEX);
  if (index == -1) {
    index = 0;
  }
  if (CUDA_CONTEXT_CACHE.count(index) == 0) {
    AVBufferRef* p = nullptr;
    std::cerr << "CREATING CUDA CONTEXT" << std::endl;

    AVDictionary* opt = nullptr;
    av_dict_set(&opt, "primary_ctx", "1", 0);

    at::DynamicLibrary cuda{"libcuda.so.1"};

    cuDeviceGet_ = (decltype(cuDeviceGet_))cuda.sym("cuDeviceGet");
    cuDevicePrimaryCtxGetState_ = (decltype(cuDevicePrimaryCtxGetState_))cuda.sym("cuDevicePrimaryCtxGetState");

    CUdevice device;
    CUDA_CHECK(cuDeviceGet_(&device, index));
    
    int dev_active = 0;
    unsigned int dev_flags = 0;
    CUDA_CHECK(cuDevicePrimaryCtxGetState_(device, &dev_flags, &dev_active));

    int ret = av_hwdevice_ctx_create(
        &p, AV_HWDEVICE_TYPE_CUDA, std::to_string(index).c_str(), opt, dev_flags);
    TORCH_CHECK(
        ret >= 0,
        "Failed to create CUDA device context on device ",
        index,
        "(",
        av_err2string(ret),
        ")");
    assert(p);
    CUDA_CONTEXT_CACHE.emplace(index, p);

    AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)p->data;
    AVCUDADeviceContext *cuda_ctx = (AVCUDADeviceContext*)device_ctx->hwctx;
    std::cerr << "CUcontext:" << cuda_ctx->cuda_ctx << std::endl;
    std::cerr << "CUstream:" << cuda_ctx->stream << std::endl;
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
