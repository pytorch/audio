#pragma once

#include <torchaudio/csrc/rnn_transducer/cpu_kernels.h>
#include <torchaudio/csrc/rnn_transducer/workspace.h>

namespace torchaudio {
namespace transducer {
namespace cpu {

// Inputs:
//   workspace: workspace.
//   logits: pointer to (B, maxT, maxU, D) logits.
//   targets: pointer to (B, maxU - 1) targets in the batch.
//   srcLengths: pointer to (B, ) source lengths in the batch.
//   tgtLengths: pointer to (B, ) target lengths in the batch.
//
// Outputs:
//   costs: pointer to (B, ) costs in the batch.
//   gradients: pointer to (B, maxT, maxU, D) gradients in the batch.
template <typename DTYPE, typename CAST_DTYPE>
status_t Compute(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* gradients = nullptr,
    const int* wpEnds = nullptr) {
  const Options& options = workspace.GetOptions();

  CHECK_EQ(options.device_, CPU);

  const int& B = options.batchSize_;
  const int& maxT = options.maxSrcLen_;
  const int& maxU = options.maxTgtLen_;
  const int& D = options.numTargets_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*N=*/B * maxT * maxU,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());
  }

  { // compute log prob pairs.
    status_t status = ComputeLogProbs<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs());
  }

  { // compute alphas and betas.
    ComputeAlphasBetas<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*alphas=*/workspace.GetPointerToAlphas(),
        /*betas=*/workspace.GetPointerToBetas(),
        /*costs=*/costs,
        /*wpEnds=*/wpEnds);
  }

  if (gradients != nullptr) {
    ComputeGradients<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*alphas=*/workspace.GetPointerToAlphas(),
        /*betas=*/workspace.GetPointerToBetas(),
        /*gradients=*/gradients);
  }

  return SUCCESS;
}

template <typename DTYPE, typename CAST_DTYPE>
status_t ComputeAlphas(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* alphas,
    const int* wpEnds = nullptr) {
  const Options& options = workspace.GetOptions();

  CHECK_EQ(options.device_, CPU);

  const int& B = options.batchSize_;
  const int& maxT = options.maxSrcLen_;
  const int& maxU = options.maxTgtLen_;
  const int& D = options.numTargets_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*N=*/B * maxT * maxU,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());
  }

  { // compute log prob pairs.
    status_t status = ComputeLogProbs<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs());
  }

  { // compute alphas.
    ComputeAlphas<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*alphas=*/alphas,
        /*wpEnds=*/wpEnds);
  }

  return SUCCESS;
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
    const int* wpEnds = nullptr) {
  const Options& options = workspace.GetOptions();

  CHECK_EQ(options.device_, CPU);

  const int& B = options.batchSize_;
  const int& maxT = options.maxSrcLen_;
  const int& maxU = options.maxTgtLen_;
  const int& D = options.numTargets_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*N=*/B * maxT * maxU,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());
  }

  { // compute log prob pairs.
    status_t status = ComputeLogProbs<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs());
  }

  { // compute betas.
    ComputeBetas<DTYPE, CAST_DTYPE>(
        /*options=*/options,
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*costs=*/costs,
        /*betas=*/betas,
        /*wpEnds=*/wpEnds);
  }

  return SUCCESS;
}

} // namespace cpu
} // namespace transducer
} // namespace torchaudio
