#pragma once

#include <torchaudio/csrc/rnnt/cpu/cpu_transducer.h>
#include <torchaudio/csrc/rnnt/gpu/gpu_transducer.h>

namespace torchaudio {
namespace rnnt {

template <typename DTYPE, typename CAST_DTYPE>
status_t Compute(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* gradients = nullptr) {
  switch (workspace.GetOptions().device_) {
    case CPU: {
      status_t status = cpu::Compute<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*gradients=*/gradients);
      return status;
    }
    case GPU: {
      status_t status = gpu::Compute<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*gradients=*/gradients);
      return status;
    }
    default: {
      return FAILURE;
    }
  };
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeAlphas(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* alphas) {
  switch (workspace.GetOptions().device_) {
    case CPU: {
      status_t status = cpu::ComputeAlphas<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*alphas=*/alphas);
      return status;
    }
    case GPU: {
      status_t status = gpu::ComputeAlphas<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/alphas);
      return status;
    }
    default: {
      return FAILURE;
    }
  };
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeBetas(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* betas) {
  switch (workspace.GetOptions().device_) {
    case CPU: {
      status_t status = cpu::ComputeBetas<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*betas=*/betas);
      return status;
    }
    case GPU: {
      status_t status = gpu::ComputeBetas<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*betas=*/betas);
      return status;
    }
    default: {
      return FAILURE;
    }
  };
}

} // namespace rnnt
} // namespace torchaudio
