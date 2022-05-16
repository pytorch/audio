#pragma once

#ifdef USE_CUDA

#include <torchaudio/csrc/rnnt/workspace.h>
#ifdef __HIP_PLATFORM_AMD__
#include <torchaudio/csrc/rnnt/hip/gpu_kernel_utils_hip.cuh>
#include <torchaudio/csrc/rnnt/hip/gpu_kernels_hip.cuh>
#else
#include <torchaudio/csrc/rnnt/gpu/gpu_kernel_utils.cuh>
#include <torchaudio/csrc/rnnt/gpu/gpu_kernels.cuh>
#endif

namespace torchaudio {
namespace rnnt {
namespace gpu {

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(
    cudaError_t code,
    const char* file,
    int line,
    bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
        stderr,
        "\nGPUassert: %s %s %d\n",
        cudaGetErrorString(code),
        file,
        line);
    if (abort)
      exit(code);
  }
}

template <typename DTYPE, typename CAST_DTYPE>
status_t LogSumExp2D(
    cudaStream_t stream,
    int N,
    int D,
    const DTYPE* logits, // [N, D]
    CAST_DTYPE* outputs) {
  { // compute max among D.
    dim3 block_dims(N);
    dim3 thread_dims(REDUCE_THREADS);

    ReduceMax2D<REDUCE_THREADS, DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*dim=*/D,
            /*inputs=*/logits,
            /*outputs=*/outputs);

    // BUGBUG: These error codes are only accurate when launching with
    // blocking. Otherwise they usually reflect earlier errors.
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_DENOMINATOR_REDUCE_MAX_FAILED;
    }
  }

  { // compute log(sum(exp(d_i - max)))
    dim3 block_dims(N);
    dim3 thread_dims(REDUCE_THREADS);

    ReduceLogSumExpGivenMax2D<REDUCE_THREADS, DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*dim=*/D,
            /*inputs=*/logits,
            /*outputs=*/outputs);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_DENOMINATOR_REDUCE_SUM_FAILED;
    }
  }

  return SUCCESS;
}

// Inputs:
//   workspace: workspace.
//   logits: pointer to (B, max_T, max_U, D) logits.
//   targets: pointer to (B, max_U - 1) targets in the batch.
//   srcLengths: pointer to (B, ) source lengths in the batch.
//   tgtLengths: pointer to (B, ) target lengths in the batch.
//
// Outputs:
//   costs: pointer to (B, ) costs in the batch.
//   gradients: pointer to (B, max_T, max_U, D) gradients in the batch.
template <typename DTYPE, typename CAST_DTYPE>
status_t Compute(
    const Workspace<CAST_DTYPE>& workspace,
    const DTYPE* logits,
    const int* targets,
    const int* srcLengths,
    const int* tgtLengths,
    DTYPE* costs,
    DTYPE* gradients = nullptr) {
  const Options& options = workspace.GetOptions();

  const cudaStream_t& stream = options.stream_;
  const int& B = options.batchSize_;
  const int& H = options.nHypos_;
  const int& max_T = options.maxSrcLen_;
  const int& max_U = options.maxTgtLen_;
  const int& D = options.numTargets_;
  const int& blank = options.blank_;
  const CAST_DTYPE clamp = options.clamp_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*stream=*/stream,
        /*N=*/B * H * max_T * max_U,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());

    if (status != SUCCESS) {
      return status;
    }
  }

  { // compute log probability pairs (blank and target).
    int num_segments =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_segments, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    ComputeLogProbs<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_LOG_PROBS_FAILED;
    }
  }

  { // compute alphas, betas and costs.
    // warp is usually a group of threads (32)
    int num_warps = (max_T + WARP_SIZE - 1) / WARP_SIZE;

    // each block is identified by 3 d tuple.
    // we are using num_warp * max_U * B * H blocks
    // where num_warp is division among Time axis
    dim3 block_dims(num_warps, max_U, B * H);

    // each thread is identified by a 2 d tuple
    // 2nd dim is 2. 1 for alpha, 1 for beta
    dim3 thread_dims(WARP_SIZE, 2);

    ComputeAlphasBetasCosts<DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*max_src_len=*/max_T,
            /*max_tgt_len=*/max_U,
            /*num_targets=*/D,
            /*blank=*/blank,
            /*log_probs=*/workspace.GetPointerToLogProbs(),
            /*srcLengths=*/srcLengths,
            /*tgtLengths=*/tgtLengths,
            /*alpha_counters=*/workspace.GetPointerToAlphaCounters(),
            /*alphas=*/workspace.GetPointerToAlphas(),
            /*beta_counters=*/workspace.GetPointerToBetaCounters(),
            /*betas=*/workspace.GetPointerToBetas(),
            /*costs=*/costs,
            /*warp_size=*/WARP_SIZE,
            /*num_warps=*/num_warps,
            H);
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_ALPHAS_BETAS_COSTS_FAILED;
    }
  }

  if (gradients != nullptr) { // compute gradients.
    // don't set gradients to zero to here as gradients might reuse memory from
    // logits

    int num_blocks =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_blocks, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    ComputeGradients<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*clamp=*/clamp,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*alphas=*/workspace.GetPointerToAlphas(),
        /*betas=*/workspace.GetPointerToBetas(),
        /*gradients=*/gradients,
        H);
    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_GRADIENTS_FAILED;
    }
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
    DTYPE* alphas) {
  const Options& options = workspace.GetOptions();

  const cudaStream_t& stream = options.stream_;
  const int& B = options.batchSize_;
  const int& H = options.nHypos_;
  const int& max_T = options.maxSrcLen_;
  const int& max_U = options.maxTgtLen_;
  const int& D = options.numTargets_;
  const int& blank = options.blank_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*stream=*/stream,
        /*N=*/B * H * max_T * max_U,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());

    if (status != SUCCESS) {
      return status;
    }
  }

  { // compute log probability pairs (blank and target).
    int num_segments =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_segments, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    ComputeLogProbs<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_LOG_PROBS_FAILED;
    }
  }
  { // compute alphas
    // warp is usually a group of threads (32)
    int num_warps = (max_T + WARP_SIZE - 1) / WARP_SIZE;

    // each block is identified by 3 d tuple.
    // we are using num_warp * max_U * B blocks
    // where num_warp is division among Time axis
    dim3 block_dims(num_warps, max_U, B * H);

    // each thread is identified by a 2 d tuple
    // 2nd dim is 1 for alpha only
    dim3 thread_dims(WARP_SIZE, 1);

    ComputeAlphasWrapper<DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*max_src_len=*/max_T,
            /*max_tgt_len=*/max_U,
            /*num_targets=*/D,
            /*blank=*/blank,
            /*log_probs=*/workspace.GetPointerToLogProbs(),
            /*srcLengths=*/srcLengths,
            /*tgtLengths=*/tgtLengths,
            /*alpha_counters=*/workspace.GetPointerToAlphaCounters(),
            /*alphas=*/(volatile DTYPE*)alphas,
            H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_ALPHAS_BETAS_COSTS_FAILED;
    }
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
    DTYPE* betas) {
  const Options& options = workspace.GetOptions();

  const cudaStream_t& stream = options.stream_;
  const int& B = options.batchSize_;
  const int& H = options.nHypos_;
  const int& max_T = options.maxSrcLen_;
  const int& max_U = options.maxTgtLen_;
  const int& D = options.numTargets_;
  const int& blank = options.blank_;

  { // compute denominators.
    status_t status = LogSumExp2D<DTYPE, CAST_DTYPE>(
        /*stream=*/stream,
        /*N=*/B * H * max_T * max_U,
        /*D=*/D,
        /*logits=*/logits,
        /*denominators=*/workspace.GetPointerToDenominators());

    if (status != SUCCESS) {
      return status;
    }
  }

  { // compute log probability pairs (blank and target).
    int num_segments =
        (max_T + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    dim3 block_dims(num_segments, max_U, B * H);
    dim3 thread_dims(MAX_THREADS_PER_BLOCK);

    ComputeLogProbs<DTYPE, CAST_DTYPE><<<block_dims, thread_dims, 0, stream>>>(
        /*max_src_len=*/max_T,
        /*max_tgt_len=*/max_U,
        /*num_targets=*/D,
        /*blank=*/blank,
        /*logits=*/logits,
        /*targets=*/targets,
        /*srcLengths=*/srcLengths,
        /*tgtLengths=*/tgtLengths,
        /*denominators=*/workspace.GetPointerToDenominators(),
        /*log_probs=*/workspace.GetPointerToLogProbs(),
        H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_LOG_PROBS_FAILED;
    }
  }
  { // compute betas
    // warp is usually a group of threads (32)
    int num_warps = (max_T + WARP_SIZE - 1) / WARP_SIZE;

    // each block is identified by 3 d tuple.
    // we are using num_warp * max_U * B blocks
    // where num_warp is division among Time axis
    dim3 block_dims(num_warps, max_U, B * H);

    // each thread is identified by a 2 d tuple
    // 2nd dim is 1 for betas only
    dim3 thread_dims(WARP_SIZE, 1);

    ComputeBetasWrapper<DTYPE, CAST_DTYPE>
        <<<block_dims, thread_dims, 0, stream>>>(
            /*max_src_len=*/max_T,
            /*max_tgt_len=*/max_U,
            /*num_targets=*/D,
            /*blank=*/blank,
            /*log_probs=*/workspace.GetPointerToLogProbs(),
            /*srcLengths=*/srcLengths,
            /*tgtLengths=*/tgtLengths,
            /*alpha_counters=*/workspace.GetPointerToBetaCounters(),
            /*alphas=*/(volatile DTYPE*)betas,
            costs,
            H);

    if (cudaGetLastError() != cudaSuccess) {
      return COMPUTE_ALPHAS_BETAS_COSTS_FAILED;
    }
  }

  return SUCCESS;
}

} // namespace gpu
} // namespace rnnt
} // namespace torchaudio

#endif // USE_CUDA
