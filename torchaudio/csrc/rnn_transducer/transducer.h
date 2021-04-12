#pragma once

#include <torchaudio/csrc/rnn_transducer/cpu_transducer.h>

namespace torchaudio {
namespace transducer {

template <typename DTYPE, typename CAST_DTYPE>
status_t Compute(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* gradients = nullptr) {
  CHECK_EQ(workspace.GetOptions().device_, CPU);
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

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeAlphas(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* alphas) {
  CHECK_EQ(workspace.GetOptions().device_, CPU);

  status_t status = cpu::ComputeAlphas<DTYPE, CAST_DTYPE>(
      /*workspace=*/workspace,
      /*logits=*/logits,
      /*targets=*/targets,
      /*srcLengths=*/srcLengths,
      /*tgtLengths=*/tgtLengths,
      /*alphas=*/alphas);
  return status;

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
  CHECK_EQ(workspace.GetOptions().device_, CPU);
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

} // namespace transducer
} // namespace torchaudio
