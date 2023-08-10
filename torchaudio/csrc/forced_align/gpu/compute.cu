#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <limits.h>
#include <torch/torch.h>
#include <cub/cub.cuh>

using namespace torch::indexing;
namespace {
constexpr int kNumThreads =
    1024; // Number of threads to run CUDA kernel in parallel.
constexpr int kBackPtrBufferSize =
    100; // Buffer size of backPtr on GPU. The data is transferred to CPU once
         // the buffer reaches this max size.
} // anonymous namespace
namespace torchaudio {
namespace alignment {
namespace gpu {
template <typename scalar_t, typename target_t>
__global__ void falign_cuda_step_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        logProbs_a,
    const torch::PackedTensorAccessor32<target_t, 2, torch::RestrictPtrTraits>
        targets_a,
    const int T,
    const int L,
    const int N,
    const int R,
    const int t,
    const int64_t blank,
    int start,
    int end,
    int backPtrBufferLen,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        alphas_a,
    torch::PackedTensorAccessor32<int8_t, 2, torch::RestrictPtrTraits>
        backPtrBuffer_a) {
  scalar_t kNegInfinity = -std::numeric_limits<scalar_t>::infinity();
  const int batchIndex =
      0; // TODO: support batch version and use the real batch index
  int S = 2 * L + 1;
  int curIdxOffset = (t % 2); // current time step frame for alpha
  int prevIdxOffset = ((t - 1) % 2); // previous time step frame for alpha
  // reset alpha and backPtrBuffer values
  for (unsigned int i = threadIdx.x; i < S; i += blockDim.x) {
    alphas_a[curIdxOffset][i] = kNegInfinity;
    backPtrBuffer_a[backPtrBufferLen][i] = -1;
  }
  // This sync could potentially be removed through careful indexing inside each
  // thread for the above for loop. But this is okay for now.
  __syncthreads();
  if (t == 0) {
    for (unsigned int i = start + threadIdx.x; i < end; i += blockDim.x) {
      int labelIdx = (i % 2 == 0) ? blank : targets_a[batchIndex][i / 2];
      alphas_a[curIdxOffset][i] = logProbs_a[batchIndex][0][labelIdx];
    }
    return;
  }
  using BlockReduce = cub::BlockReduce<scalar_t, kNumThreads>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ scalar_t maxValue;
  scalar_t threadMax;
  int startloop = start;
  threadMax = kNegInfinity;
  if (start == 0 && threadIdx.x == 0) {
    alphas_a[curIdxOffset][0] =
        alphas_a[prevIdxOffset][0] + logProbs_a[batchIndex][t][blank];
    threadMax = max(threadMax, alphas_a[curIdxOffset][0]);
    backPtrBuffer_a[backPtrBufferLen][0] = 0;
  }
  if (start == 0) {
    startloop += 1;
  }
  for (unsigned int i = startloop + threadIdx.x; i < end; i += blockDim.x) {
    scalar_t x0 = alphas_a[prevIdxOffset][i];
    scalar_t x1 = alphas_a[prevIdxOffset][i - 1];
    scalar_t x2 = kNegInfinity;
    int labelIdx = (i % 2 == 0) ? blank : targets_a[batchIndex][i / 2];
    if (i % 2 != 0 && i != 1 &&
        targets_a[batchIndex][i / 2] != targets_a[batchIndex][i / 2 - 1]) {
      x2 = alphas_a[prevIdxOffset][i - 2];
    }
    scalar_t result = 0.0;
    if (x2 > x1 && x2 > x0) {
      result = x2;
      backPtrBuffer_a[backPtrBufferLen][i] = 2;
    } else if (x1 > x0 && x1 > x2) {
      result = x1;
      backPtrBuffer_a[backPtrBufferLen][i] = 1;
    } else {
      result = x0;
      backPtrBuffer_a[backPtrBufferLen][i] = 0;
    }
    alphas_a[curIdxOffset][i] = result + logProbs_a[batchIndex][t][labelIdx];
    threadMax = max(threadMax, alphas_a[curIdxOffset][i]);
  }
  scalar_t maxResult = BlockReduce(tempStorage).Reduce(threadMax, cub::Max());
  if (threadIdx.x == 0) {
    maxValue = maxResult;
  }
  __syncthreads();
  // normalize alpha values so that they don't overflow for large T
  for (unsigned int i = threadIdx.x; i < S; i += blockDim.x) {
    alphas_a[curIdxOffset][i] -= maxValue;
  }
}

template <typename scalar_t, torch::ScalarType target_scalar_type>
void forced_align_impl(
    const torch::Tensor& logProbs,
    const torch::Tensor& targets,
    const int64_t blank,
    torch::Tensor& paths) {
  auto defaultStream = at::cuda::getCurrentCUDAStream();
  auto cpuDataTranferStream = at::cuda::getStreamFromPool();
  const scalar_t kNegInfinity = -std::numeric_limits<scalar_t>::infinity();
  using target_t = typename std::
      conditional<target_scalar_type == torch::kInt, int, int64_t>::type;
  auto paths_a = paths.accessor<target_t, 2>();
  const int batchIndex =
      0; // TODO: support batch version and use the real batch index
  const int T = logProbs.size(1); // num frames
  const int N = logProbs.size(2); // alphabet size
  const int L = targets.size(1); // label length
  const int S = 2 * L + 1;
  auto targetsCpu = targets.to(torch::kCPU);
  // backPtrBuffer stores the index offset fthe best path at current position
  // We copy the values to CPU after running every kBackPtrBufferSize of
  // frames.
  torch::Tensor backPtrBuffer =
      torch::empty(
          {min(kBackPtrBufferSize, T), S},
          torch::TensorOptions().dtype(torch::kInt8).device(logProbs.device()))
          .contiguous()
          .fill_(-1);
  torch::Tensor backPtrCpu =
      torch::empty(
          {T, S},
          torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU))
          .contiguous()
          .fill_(-1);
  // we store only two time frames for alphas
  // alphas for compute current timeframe can be computed only from previous
  // time frame.
  torch::Tensor alphas = torch::empty(
                             {2, S},
                             torch::TensorOptions()
                                 .dtype(logProbs.dtype())
                                 .device(logProbs.device()))
                             .fill_(kNegInfinity);
  // CPU accessors
  auto targetsCpu_a = targetsCpu.accessor<target_t, 2>();
  auto backPtrCpu_a = backPtrCpu.accessor<int8_t, 2>();
  // count the number of repeats in label
  int R = 0;
  for (int i = 1; i < L; ++i) {
    if (targetsCpu_a[batchIndex][i] == targetsCpu_a[batchIndex][i - 1]) {
      ++R;
    }
  }
  TORCH_CHECK(
      T >= L + R,
      "targets length is too long for CTC. Found targets length: ",
      T,
      ", log_probs length: ",
      L,
      ", and number of repeats: ",
      R);
  int start = (T - (L + R)) > 0 ? 0 : 1;
  int end = (S == 1) ? 1 : 2;
  int backPtrBufferLen = 0;
  torch::Tensor bufferCopy;
  for (int t = 0; t < T; ++t) {
    if (t > 0) {
      if (T - t <= L + R) {
        if ((start % 2 == 1) &&
            (targetsCpu_a[batchIndex][start / 2] !=
             targetsCpu_a[batchIndex][start / 2 + 1])) {
          start = start + 1;
        }
        start = start + 1;
      }
      if (t <= L + R) {
        if ((end % 2 == 0) && (end < 2 * L) &&
            (targetsCpu_a[batchIndex][end / 2 - 1] !=
             targetsCpu_a[batchIndex][end / 2])) {
          end = end + 1;
        }
        end = end + 1;
      }
    }
    falign_cuda_step_kernel<scalar_t, target_t>
        <<<1, kNumThreads, 0, defaultStream>>>(
            logProbs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            targets.packed_accessor32<target_t, 2, torch::RestrictPtrTraits>(),
            T,
            L,
            N,
            R,
            t,
            blank,
            start,
            end,
            backPtrBufferLen,
            alphas.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            backPtrBuffer
                .packed_accessor32<int8_t, 2, torch::RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ++backPtrBufferLen;
    if (backPtrBufferLen == kBackPtrBufferSize || t == T - 1) {
      cpuDataTranferStream.synchronize();
      // GPU -> GPU copy
      bufferCopy = backPtrBuffer.clone().contiguous();
      defaultStream.synchronize();
      at::cuda::setCurrentCUDAStream(cpuDataTranferStream);
      // Copy ASYNC from GPU to CPU
      int64_t offset =
          static_cast<int64_t>(t + 1 - backPtrBufferLen) * S * sizeof(int8_t);
      C10_CUDA_CHECK(cudaMemcpyAsync(
          static_cast<int8_t*>(backPtrCpu.data_ptr()) + offset,
          bufferCopy.data_ptr(),
          backPtrBufferLen * S * sizeof(int8_t),
          cudaMemcpyDeviceToHost,
          cpuDataTranferStream));
      at::cuda::setCurrentCUDAStream(defaultStream);
      backPtrBufferLen = 0;
    }
  }
  cpuDataTranferStream.synchronize();
  torch::Tensor alphasCpu = alphas.to(torch::kCPU);
  auto alphasCpu_a = alphasCpu.accessor<scalar_t, 2>();
  int curIdxOffset = ((T - 1) % 2);
  int ltrIdx =
      alphasCpu_a[curIdxOffset][S - 1] > alphasCpu_a[curIdxOffset][S - 2]
      ? S - 1
      : S - 2;
  int indexScores = 0;
  for (int t = T - 1; t >= 0; --t) {
    auto lbl_idx =
        ltrIdx % 2 == 0 ? blank : targetsCpu_a[batchIndex][ltrIdx / 2];
    paths_a[batchIndex][t] = lbl_idx;
    ++indexScores;
    ltrIdx -= backPtrCpu_a[t][ltrIdx];
  }
}

std::tuple<torch::Tensor, torch::Tensor> compute(
    const torch::Tensor& logProbs,
    const torch::Tensor& targets,
    const torch::Tensor& inputLengths,
    const torch::Tensor& targetLengths,
    const int64_t blank) {
  TORCH_CHECK(logProbs.is_cuda(), "log_probs must be a CUDA tensor");
  TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
  TORCH_CHECK(
      logProbs.device() == targets.device(),
      "log_probs and targets need to be on the same device");
  TORCH_CHECK(
      logProbs.dtype() == torch::kFloat64 ||
          logProbs.dtype() == torch::kFloat32 ||
          logProbs.dtype() == torch::kFloat16,
      "log_probs must be float64, float32 or float16 (half) type");
  TORCH_CHECK(
      targets.dtype() == torch::kInt32 || targets.dtype() == torch::kInt64,
      "targets must be int32 or int64 type");
  TORCH_CHECK(logProbs.is_contiguous(), "log_probs must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  TORCH_CHECK(
      logProbs.dim() == 3,
      "log_probs must be 3-D (batch_size, input length, num classes)");
  TORCH_CHECK(
      targets.dim() == 2, "targets must be 2-D (batch_size, target length,)");
  TORCH_CHECK(
      inputLengths.dim() == 1, "input_lengths must be 1-D (batch_size,)");
  TORCH_CHECK(
      targetLengths.dim() == 1, "target_lengths must be 1-D (batch_size,)");
  TORCH_CHECK(
      logProbs.size(0) == 1,
      "The batch dimension for log_probs must be 1 at the current version.")
  TORCH_CHECK(
      targets.size(0) == 1,
      "The batch dimension for targets must be 1 at the current version.")
  TORCH_CHECK(
      blank >= 0 && blank < logProbs.size(-1),
      "blank must be within [0, num classes)");

  TORCH_CHECK(
      logProbs.size(1) == at::max(inputLengths).item().toInt(),
      "input length mismatch");
  TORCH_CHECK(
      targets.size(1) == at::max(targetLengths).item().toInt(),
      "target length mismatch");

  auto B = logProbs.size(0);
  auto T = logProbs.size(1); // num frames
  auto paths = torch::zeros(
      {B, T},
      torch::TensorOptions().device(torch::kCPU).dtype(targets.dtype()));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      logProbs.scalar_type(), "forced_align_impl", [&] {
        if (targets.scalar_type() == torch::kInt64) {
          forced_align_impl<scalar_t, torch::kInt64>(
              logProbs, targets, blank, paths);
        } else {
          forced_align_impl<scalar_t, torch::kInt32>(
              logProbs, targets, blank, paths);
        }
      });
  return std::make_tuple(
      paths.to(logProbs.device()),
      logProbs.index(
          {torch::indexing::Slice(),
           torch::linspace(
               0, T - 1, T, torch::TensorOptions().dtype(paths.dtype())),
           paths.index({0})}));
}

TORCH_LIBRARY_IMPL(torchaudio, CUDA, m) {
  m.impl("forced_align", &compute);
}

} // namespace gpu
} // namespace alignment
} // namespace torchaudio
