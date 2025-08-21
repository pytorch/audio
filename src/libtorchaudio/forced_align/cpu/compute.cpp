#include <c10/util/Exception.h>
#include <libtorchaudio/accessor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
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

  auto alphas_a = new scalar_t[2 * S]; // scalar_t is just logProbs.dtype()
  for (int i = 0; i < 2 * S; i++) {
    alphas_a[i] = kNegInfinity;
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
  AOTI_TORCH_CHECK(
      T >= L + R,
      "targets length is too long for CTC");
  auto start = T - (L + R) > 0 ? 0 : 1;
  auto end = (S == 1) ? 1 : 2;
  for (auto i = start; i < end; i++) {
    auto labelIdx = (i % 2 == 0) ? blank : targets_a.index(batchIndex, i / 2);
    alphas_a[i] = logProbs_a.index(batchIndex,0,labelIdx);

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
      alphas_a[curIdxOffset * S + j] = -std::numeric_limits<scalar_t>::infinity(); // alphas_a[curIdxOffset][j]
    }
    if (start == 0) {
      alphas_a[curIdxOffset * S] =
          alphas_a[prevIdxOffset * S] + logProbs_a.index(batchIndex, t, blank);
      backPtr_a[S * t] = 0;  // backPtr_a[t][0] = 0
      startloop += 1;
    }

    for (auto i = startloop; i < end; i++) {
      auto x0 = alphas_a[prevIdxOffset * S + i]; // alphas_a[prevIdxOffset][i];
      auto x1 = alphas_a[prevIdxOffset * S + i - 1]; // alphas_a[prevIdxOffset][i - 1];
      auto x2 = -std::numeric_limits<scalar_t>::infinity();

      auto labelIdx = (i % 2 == 0) ? blank : targets_a.index(batchIndex, i / 2);

      // In CTC, the optimal path may optionally chose to skip a blank label.
      // x2 represents skipping a letter, and can only happen if we're not
      // currently on a blank_label, and we're not on a repeat letter
      // (i != 1) just ensures we don't access targets[i - 2] if its i < 2
      if (i % 2 != 0 && i != 1 &&
          targets_a.index(batchIndex, i / 2) != targets_a.index(batchIndex, i / 2 - 1)) {
        x2 = alphas_a[prevIdxOffset * S + i - 2]; // alphas_a[prevIdxOffset][i - 2];
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

      alphas_a[curIdxOffset * S + i] = result + logProbs_a.index(batchIndex, t, labelIdx); // alphas_a[curIdxOffset][i]
    }
  }
  auto idx1 = (T - 1) % 2;
  auto ltrIdx = alphas_a[S * idx1 + S - 1] >
    alphas_a[S * idx1 + S - 2] ? S - 1 : S - 2; // alphas_a[idx1][S - 1], alphas_a[idx1][S - 2]
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
    const Tensor logProbs,
    const Tensor targets,
    Tensor inputLengths,
    Tensor targetLengths,
    const int64_t blank) {
  AOTI_TORCH_CHECK(logProbs.is_cpu(), "log_probs must be a CPU tensor");
  AOTI_TORCH_CHECK(targets.is_cpu(), "targets must be a CPU tensor");
  AOTI_TORCH_CHECK(
      logProbs.get_device() == targets.get_device(),
      "log_probs and targets need to be on the same device");
  int32_t logprobs_dtype;
  aoti_torch_get_dtype(logProbs.get(), &logprobs_dtype);
  AOTI_TORCH_CHECK(
    logprobs_dtype == aoti_torch_dtype_float64() ||
    logprobs_dtype == aoti_torch_dtype_float32() ||
    logprobs_dtype == aoti_torch_dtype_float16(),
      "log_probs must be float64, float32 or float16 (half) type");
  int32_t targets_dtype;
  aoti_torch_get_dtype(targets.get(), &targets_dtype);
  AOTI_TORCH_CHECK(
    targets_dtype == aoti_torch_dtype_int32() || targets_dtype == aoti_torch_dtype_int64(),
      "targets must be int32 or int64 type");
  AOTI_TORCH_CHECK(logProbs.is_contiguous(), "log_probs must be contiguous");
  AOTI_TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  AOTI_TORCH_CHECK(
      logProbs.dim() == 3,
      "log_probs must be 3-D (batch_size, input length, num classes)");
  AOTI_TORCH_CHECK(
      targets.dim() == 2, "targets must be 2-D (batch_size, target length,)");
  AOTI_TORCH_CHECK(
      inputLengths.dim() == 1, "input_lengths must be 1-D (batch_size,)");
  AOTI_TORCH_CHECK(
      targetLengths.dim() == 1, "target_lengths must be 1-D (batch_size,)");
  AOTI_TORCH_CHECK(
      logProbs.size(0) == 1,
      "The batch dimension for log_probs must be 1 at the current version.")
  AOTI_TORCH_CHECK(
      targets.size(0) == 1,
      "The batch dimension for targets must be 1 at the current version.")
  AOTI_TORCH_CHECK(
      blank >= 0 && blank < logProbs.size(-1),
      "blank must be within [0, num classes)");

  int32_t targetLengths_dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(targetLengths.get(), &targetLengths_dtype));
  AOTI_TORCH_CHECK(
    targetLengths_dtype == aoti_torch_dtype_int32() || targetLengths_dtype == aoti_torch_dtype_int64(),
      "target lengths must be int32 or int64 type");
  auto target_length_max = amax(targetLengths, 0, false);
  void *target_max_length_ptr = target_length_max.data_ptr();
  int64_t target_max_length;
  if (targetLengths_dtype == aoti_torch_dtype_int32()) {
    printf("\n\n## INT32\n\n");
    int32_t *ptr = (int32_t *)(target_max_length_ptr);
    target_max_length = (int64_t)(*ptr);
  } else if (targetLengths_dtype == aoti_torch_dtype_int64()) {
    printf("\n\n## INT64\n\n");
    target_max_length = *((int64_t *)(target_max_length_ptr));
  }
  printf("TARGET MAX LENGTH IS %ld\n", target_max_length);
  TORCH_CHECK(targets.size(1) == target_max_length, "target length mismatch");

  int32_t inputLengths_dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(inputLengths.get(), &inputLengths_dtype));
  AOTI_TORCH_CHECK(
    inputLengths_dtype == aoti_torch_dtype_int32() || inputLengths_dtype == aoti_torch_dtype_int64(),
      "input lengths must be int32 or int64 type");
  auto input_length_max =  amax(inputLengths, 0, false);
  void *input_max_length_ptr = input_length_max.data_ptr();
  int64_t input_max_length;
  if (inputLengths_dtype == aoti_torch_dtype_int32()) {
    int32_t *ptr = (int32_t *)(input_max_length_ptr);
    input_max_length = (int64_t)(*ptr);
  } else if (inputLengths_dtype == aoti_torch_dtype_int64()) {
    input_max_length = *((int64_t *)(input_max_length_ptr));
  }
  TORCH_CHECK(logProbs.size(1) == input_max_length, "input length mismatch");

  const auto B = logProbs.size(0);
  const auto T = logProbs.size(1);

  int64_t paths_size[2] = {B, T};
  int64_t paths_stride[2] = {T, 1};
  AtenTensorHandle paths_h;
  int32_t targets_device;
  aoti_torch_get_device_type(targets.get(), &targets_device);
  aoti_torch_empty_strided(2, paths_size, paths_stride, targets_dtype, targets_device, targets.get_device(), &paths_h);
  auto paths = Tensor(paths_h);


  if (targets_dtype == aoti_torch_dtype_int64()) {
    if (logprobs_dtype == aoti_torch_dtype_float64()) {
      forced_align_impl<double, int64_t>(logProbs, targets, blank, paths);
    } else if (logprobs_dtype == aoti_torch_dtype_float32()) {
      forced_align_impl<float, int64_t>(logProbs, targets, blank, paths);
    } else if (logprobs_dtype == aoti_torch_dtype_float16()) {
      forced_align_impl<c10::Half, int64_t>(logProbs, targets, blank, paths);
    }
  } else if (targets_dtype == aoti_torch_dtype_int32()) {
    if (logprobs_dtype == aoti_torch_dtype_float64()) {
      forced_align_impl<double, int32_t>(logProbs, targets, blank, paths);
    } else if (logprobs_dtype == aoti_torch_dtype_float32()) {
      forced_align_impl<float, int32_t>(logProbs, targets, blank, paths);
    } else if (logprobs_dtype == aoti_torch_dtype_float16()) {
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
