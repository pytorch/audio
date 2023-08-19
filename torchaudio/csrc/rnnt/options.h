#pragma once

//#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
typedef cudaStream_t gpuStream_t;
#endif // USE_CUDA
#ifdef USE_ROCM
#include <hip/hip_runtime.h>
typedef hipStream_t gpuStream_t;
#endif // USE_ROCM

#include <torchaudio/csrc/rnnt/macros.h>
#include <torchaudio/csrc/rnnt/types.h>

namespace torchaudio {
namespace rnnt {

typedef struct Options {
  // the device to compute transducer loss.
  device_t device_;
#if defined(USE_CUDA) || defined(USE_ROCM)
  // the stream to launch kernels in when using GPU.
  gpuStream_t stream_;
#endif
  // The maximum number of threads that can be used.
  int numThreads_;

  // the index for "blank".
  int blank_;
  // whether to backtrack the best path.
  bool backtrack_;
  // gradient clamp value.
  float clamp_;

  // batch size = B.
  int batchSize_;

  // Number of hypos per sample = H
  int nHypos_;

  // the maximum length of src encodings = max_T.
  int maxSrcLen_;
  // the maximum length of tgt encodings = max_U.
  int maxTgtLen_;
  // num_targets = D.
  int numTargets_;

  // if set to true, inputs are logits and gradients are
  // fused with logsoftmax gradients.
  // if set to false, log_softmax is computed outside of loss
  // True by default
  bool fusedLogSmax_;

  Options()
      : device_(UNDEFINED),
        numThreads_(0),
        blank_(-1),
        backtrack_(false),
        clamp_(-1), // negative for disabling clamping by default.
        batchSize_(0),
        nHypos_(1),
        maxSrcLen_(0),
        maxTgtLen_(0),
        numTargets_(0),
        fusedLogSmax_(true) {}

  int BU() const {
    return batchSize_ * maxTgtLen_ * nHypos_;
  }

  int BTU() const {
    return batchSize_ * maxSrcLen_ * maxTgtLen_ * nHypos_;
  }

  friend std::ostream& operator<<(std::ostream& os, const Options& options) {
    os << "Options("
       << "batchSize_=" << options.batchSize_ << ", "
       << "maxSrcLen_=" << options.maxSrcLen_ << ", "
       << "maxTgtLen_=" << options.maxTgtLen_ << ", "
       << "numTargets_=" << options.numTargets_ << ")";

    return os;
  }
} Options;

} // namespace rnnt
} // namespace torchaudio
