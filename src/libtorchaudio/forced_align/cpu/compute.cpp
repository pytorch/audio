#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <libtorchaudio/accessor.h>
#include <torch/headeronly/util/Half.h>


using namespace std;

namespace torchaudio {
namespace alignment {
namespace cpu {

using torch::stable::Tensor;

// Inspired from
// https://github.com/flashlight/sequence/blob/main/flashlight/lib/sequence/criterion/cpu/ConnectionistTemporalClassificationCriterion.cpp
template <typename scalar_t, typename target_t>
void forced_align_impl(
    const Tensor logProbs,
    const Tensor targets,
    target_t blank,
    Tensor paths) {
  const scalar_t kNegInfinity = -std::numeric_limits<scalar_t>::infinity();
  const auto batchIndex =
      0; // TODO: support batch version and use the real batch index
  const auto T = logProbs.size(1);
  const auto L = targets.size(1);
  const auto S = 2 * L + 1;

  auto alphas_a = new scalar_t[S][2]; // scalar_t is just logProbs.dtype()
  for (int i = 0; i < S; i++) {
    alphas_a[i][0] = kNegInfinity;
    alphas_a[i][1] = kNegInfinity;
  }

  auto backPtr_a = new int8_t[T * S];
  for (int i = 0; i < T * S; i++) {
    backPtr_a[i] = -1;
  }

  auto logProbs_a = Accessor<3, scalar_t, true>(logProbs);
  auto targets_a = Accessor<2, target_t, true>(targets);
  auto paths_a = Accessor<2, target_t, false>(paths);
  auto R = 0;
  for (auto i = 1; i < L; i++) {
    if (targets_a.index(batchIndex, i) == targets_a.index(batchIndex, i - 1)) {
      ++R;
    }
  }
  TORCH_CHECK(
      T >= L + R,
      "targets length is too long for CTC. Found log_probs length: ",
      T,
      ", targets length: ",
      L,
      ", and number of repeats: ",
      R);
  auto start = T - (L + R) > 0 ? 0 : 1;
  auto end = (S == 1) ? 1 : 2;
  for (auto i = start; i < end; i++) {
    auto labelIdx = (i % 2 == 0) ? blank : targets_a.index(batchIndex, i / 2);
    alphas_a[i][0] = logProbs_a.index(batchIndex,0,labelIdx);
  }
  for (auto t = 1; t < T; t++) {
    if (T - t <= L + R) {
      if ((start % 2 == 1) &&
          targets_a.index(batchIndex, start / 2) !=
              targets_a.index(batchIndex, start / 2 + 1)) {
        start = start + 1;
      }
      start = start + 1;
    }
    if (t <= L + R) {
      if (end % 2 == 0 && end < 2 * L &&
          targets_a.index(batchIndex, end / 2 - 1) !=
              targets_a.index(batchIndex, end / 2)) {
        end = end + 1;
      }
      end = end + 1;
    }
    auto startloop = start;
    auto curIdxOffset = t % 2;
    auto prevIdxOffset = (t - 1) % 2;
    for (auto j = 0; j < S; ++j) {
      alphas_a[j][curIdxOffset] = -std::numeric_limits<scalar_t>::infinity();
    }
    if (start == 0) {
      alphas_a[0][curIdxOffset] =
          alphas_a[0][prevIdxOffset] + logProbs_a.index(batchIndex, t, blank);
      backPtr_a[S * t] = 0;  // backPtr_a[t][0] = 0
      startloop += 1;
    }

    for (auto i = startloop; i < end; i++) {
      auto x0 = alphas_a[i][prevIdxOffset];
      auto x1 = alphas_a[i - 1][prevIdxOffset];
      auto x2 = -std::numeric_limits<scalar_t>::infinity();

      auto labelIdx = (i % 2 == 0) ? blank : targets_a.index(batchIndex, i / 2);

      // In CTC, the optimal path may optionally chose to skip a blank label.
      // x2 represents skipping a letter, and can only happen if we're not
      // currently on a blank_label, and we're not on a repeat letter
      // (i != 1) just ensures we don't access targets[i - 2] if its i < 2
      if (i % 2 != 0 && i != 1 &&
          targets_a.index(batchIndex, i / 2) != targets_a.index(batchIndex, i / 2 - 1)) {
        x2 = alphas_a[i - 2][prevIdxOffset];
      }
      scalar_t result = 0.0;
      if (x2 > x1 && x2 > x0) {
        result = x2;
        backPtr_a[t * S + i] = 2; // backPtr_a[t][i] = 2
      } else if (x1 > x0 && x1 > x2) {
        result = x1;
        backPtr_a[t * S + i] = 1; // backPtr_a[t][i] = 1
      } else {
        result = x0;
        backPtr_a[t * S + i] = 0; // backPtr_a[t][i] = 0
      }
      alphas_a[i][curIdxOffset] = result + logProbs_a.index(batchIndex, t, labelIdx);
    }
  }
  auto idx1 = (T - 1) % 2;
  auto ltrIdx = alphas_a[S - 1][idx1] > alphas_a[S - 2][idx1] ? S - 1 : S - 2;
  delete[] alphas_a;
  // path stores the token index for each time step after force alignment.
  for (auto t = T - 1; t > -1; t--) {
    auto lbl_idx = ltrIdx % 2 == 0 ? blank : targets_a.index(batchIndex, ltrIdx / 2);
    paths_a.set_index(lbl_idx, batchIndex, t);
    ltrIdx -= backPtr_a[t * S + ltrIdx]; // backPtr_a[t][ltrIdx]
  }
  delete[] backPtr_a;
}

std::tuple<Tensor, Tensor> compute(
    const Tensor& logProbs,
    const Tensor& targets,
    const Tensor& inputLengths,
    const Tensor& targetLengths,
    const int64_t blank) {
  TORCH_CHECK(logProbs.is_cpu(), "log_probs must be a CPU tensor");
  TORCH_CHECK(targets.is_cpu(), "targets must be a CPU tensor");
  TORCH_CHECK(
      logProbs.get_device() == targets.get_device(),
      "log_probs and targets need to be on the same device");
  TORCH_CHECK(
      logProbs.dtype() == aoti_torch_dtype_float64() ||
          logProbs.dtype() == aoti_torch_dtype_float32() ||
          logProbs.dtype() == aoti_torch_dtype_float16(),
      "log_probs must be float64, float32 or float16 (half) type");
  TORCH_CHECK(
      targets.dtype() == aoti_torch_dtype_int32() || targets.dtype() == aoti_torch_dtype_int64(),
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

  // TODO: Requires port of `max` operator.
  // TORCH_CHECK(
  //     logProbs.size(1) == at::max(inputLengths).item().toInt(),
  //     "input length mismatch");
  // TORCH_CHECK(
  //     targets.size(1) == at::max(targetLengths).item().toInt(),
  //     "target length mismatch");

  const auto B = logProbs.size(0);
  const auto T = logProbs.size(1);

  int64_t paths_size[2] = {B, T};
  int64_t paths_stride[2] = {T, 1};
  AtenTensorHandle paths_h;
  int32_t targets_device;
  aoti_torch_get_device_type(targets.get(), &targets_device);
  aoti_torch_empty_strided(1, paths_size, paths_stride, targets.dtype(), targets_device, targets.get_device(), &paths_h);
  auto paths = Tensor(paths_h);


  if (targets.dtype() == aoti_torch_dtype_int64()) {
    if (logProbs.dtype() == aoti_torch_dtype_float64()) {
      forced_align_impl<double, int64_t>(logProbs, targets, blank, paths);
    } else if (logProbs.dtype() == aoti_torch_dtype_float32()) {
      forced_align_impl<float, int64_t>(logProbs, targets, blank, paths);
    } else if (logProbs.dtype() == aoti_torch_dtype_float16()) {
      forced_align_impl<c10::Half, int64_t>(logProbs, targets, blank, paths);
    }
  } else if (targets.dtype() == aoti_torch_dtype_int32()) {
    if (logProbs.dtype() == aoti_torch_dtype_float64()) {
      forced_align_impl<double, int32_t>(logProbs, targets, blank, paths);
    } else if (logProbs.dtype() == aoti_torch_dtype_float32()) {
      forced_align_impl<float, int32_t>(logProbs, targets, blank, paths);
    } else if (logProbs.dtype() == aoti_torch_dtype_float16()) {
      forced_align_impl<c10::Half, int32_t>(logProbs, targets, blank, paths);
    }
  }
  return std::make_tuple(
      paths,
      logProbs
      );
}


void boxed_compute(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor t1(to<AtenTensorHandle>(stack[0]));
  Tensor t2(to<AtenTensorHandle>(stack[1]));
  Tensor t3(to<AtenTensorHandle>(stack[2]));
  Tensor t4(to<AtenTensorHandle>(stack[3]));
  int64_t blank = to<int64_t>(stack[4]);
  auto result = compute(
      std::move(t1), std::move(t2), std::move(t3), std::move(t4), blank);
  stack[0] = from(std::get<0>(result));
  stack[1] = from(std::get<1>(result));
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("forced_align", &boxed_compute);
}

} // namespace cpu
} // namespace alignment
} // namespace torchaudio
