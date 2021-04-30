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
    DTYPE* gradients = nullptr,
    const int* wp_ends=nullptr) {
  switch (workspace.GetOptions().device_) {
    case CPU: {
      status_t status = cpu::Compute<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*gradients=*/gradients,
          /*wp_ends =*/wp_ends);
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
          /*gradients=*/gradients,
          /*wp_ends =*/wp_ends);
      return status;
    }
    default: {
      LOG(ERROR) << "unsupported workspace.GetOptions().device = "
                 << workspace.GetOptions().device_;
      return FAILURE;
    }
  };
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeSparse(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* gradients = nullptr,
    const int* wp_ends=nullptr,
    const int* validRanges=nullptr,
    const int* cellsPerSample=nullptr) {
  switch (workspace.GetOptions().device_) {
    case GPU: {
      status_t status = gpu::ComputeSparse<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*gradients=*/gradients,
          /*wp_ends =*/wp_ends,
          validRanges,
          cellsPerSample);
      return status;
    }
    default: {
      LOG(ERROR) << "unsupported workspace.GetOptions().device = "
                 << workspace.GetOptions().device_;
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
    DTYPE* alphas,
    const int* wp_ends=nullptr) {
  switch (workspace.GetOptions().device_) {
    case CPU: {
      status_t status = cpu::ComputeAlphas<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*alphas=*/alphas,
          /*wp_ends =*/wp_ends);
      return status;
    }
    case GPU: {
      status_t status = gpu::ComputeAlphas<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/alphas,
          /*wp_ends =*/wp_ends);
      return status;
    }
    default: {
      LOG(ERROR) << "unsupported workspace.GetOptions().device = "
                 << workspace.GetOptions().device_;
      return FAILURE;
    }
  };
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeAlphasSparse(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* alphas,
    const int* wp_ends=nullptr,
    const int* validRanges=nullptr,
    const int* cellsPerSample=nullptr) {
  switch (workspace.GetOptions().device_) {
    case GPU: {
      status_t status = gpu::ComputeAlphasSparse<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/alphas,
          /*wp_ends =*/wp_ends,
          validRanges,
          cellsPerSample);
      return status;
    }
    default: {
      LOG(ERROR) << "unsupported workspace.GetOptions().device = "
                 << workspace.GetOptions().device_;
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
    DTYPE* betas,
    const int* wp_ends=nullptr) {
  switch (workspace.GetOptions().device_) {
    case CPU: {
      status_t status = cpu::ComputeBetas<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*betas=*/betas,
          /*wp_ends =*/wp_ends);
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
          /*betas=*/betas,
          /*wp_ends =*/wp_ends);
      return status;
    }
    default: {
      LOG(ERROR) << "unsupported workspace.GetOptions().device = "
                 << workspace.GetOptions().device_;
      return FAILURE;
    }
  };
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeBetasSparse(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* betas,
    const int* wp_ends=nullptr,
    const int* validRanges=nullptr,
    const int* cellsPerSample=nullptr) {
  switch (workspace.GetOptions().device_) {
    case GPU: {
      status_t status = gpu::ComputeBetasSparse<DTYPE, CAST_DTYPE>(
          /*workspace=*/workspace,
          /*logits=*/logits,
          /*targets=*/targets,
          /*srcLengths=*/srcLengths,
          /*tgtLengths=*/tgtLengths,
          /*costs=*/costs,
          /*betas=*/betas,
          /*wp_ends =*/wp_ends,
          validRanges,
          cellsPerSample);
      return status;
    }
    default: {
      LOG(ERROR) << "unsupported workspace.GetOptions().device = "
                 << workspace.GetOptions().device_;
      return FAILURE;
    }
  };
}

} // namespace rnnt
} // namespace torchaudio
