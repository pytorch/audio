#include <libtorchaudio/stable/ops.h>
#include <libtorchaudio/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch_v2.h>
#include <torch/headeronly/core/ScalarType.h>

namespace torchaudio {
namespace alignment {
namespace cpu {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

// Inspired from
// https://github.com/flashlight/sequence/blob/main/flashlight/lib/sequence/criterion/cpu/ConnectionistTemporalClassificationCriterion.cpp
template <typename scalar_t, ScalarType target_scalar_type>
void forced_align_impl(
    const Tensor& logProbs,
    const Tensor& targets,
    const int64_t blank,
    Tensor& paths) {
  const scalar_t kNegInfinity = -std::numeric_limits<scalar_t>::infinity();
  using target_t = typename std::
      conditional<target_scalar_type == ScalarType::Int, int, int64_t>::type;
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
  auto logProbs_a = accessor<scalar_t, 3>(logProbs);
  auto targets_a = accessor<target_t, 2>(targets);
  auto paths_a = accessor<target_t, 2>(paths);
  auto R = 0;
  for (auto i = 1; i < L; i++) {
    if (targets_a[batchIndex][i] == targets_a[batchIndex][i - 1]) {
      ++R;
    }
  }
  STD_TORCH_CHECK(
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
    auto labelIdx = (i % 2 == 0) ? blank : targets_a[batchIndex][i / 2];
    alphas_a[i] = logProbs_a[batchIndex][0][labelIdx]; // alphas_a[0, i]
  }
  for (auto t = 1; t < T; t++) {
    if (T - t <= L + R) {
      if ((start % 2 == 1) &&
          targets_a[batchIndex][start / 2] !=
              targets_a[batchIndex][start / 2 + 1]) {
        start = start + 1;
      }
      start = start + 1;
    }
    if (t <= L + R) {
      if (end % 2 == 0 && end < 2 * L &&
          targets_a[batchIndex][end / 2 - 1] !=
              targets_a[batchIndex][end / 2]) {
        end = end + 1;
      }
      end = end + 1;
    }
    auto startloop = start;
    auto curIdxOffset = t % 2;
    auto prevIdxOffset = (t - 1) % 2;
    for (auto j = 0; j < S; ++j) {
      alphas_a[curIdxOffset * S + j] = -std::numeric_limits<
          scalar_t>::infinity(); // alphas_a[curIdxOffset][j]
    }
    if (start == 0) {
      alphas_a[curIdxOffset * S] = alphas_a[prevIdxOffset * S] +
          logProbs_a[batchIndex][t][blank]; // alphas_a[curIdxOffset][0]
      backPtr_a[S * t] = 0; // backPtr_a[t][0] = 0
      startloop += 1;
    }

    for (auto i = startloop; i < end; i++) {
      auto x0 = alphas_a[prevIdxOffset * S + i]; // alphas_a[prevIdxOffset][i];
      auto x1 =
          alphas_a[prevIdxOffset * S + i - 1]; // alphas_a[prevIdxOffset][i
                                               // - 1];
      auto x2 = -std::numeric_limits<scalar_t>::infinity();

      auto labelIdx = (i % 2 == 0) ? blank : targets_a[batchIndex][i / 2];

      // In CTC, the optimal path may optionally chose to skip a blank label.
      // x2 represents skipping a letter, and can only happen if we're not
      // currently on a blank_label, and we're not on a repeat letter
      // (i != 1) just ensures we don't access targets[i - 2] if its i < 2
      if (i % 2 != 0 && i != 1 &&
          targets_a[batchIndex][i / 2] != targets_a[batchIndex][i / 2 - 1]) {
        x2 = alphas_a[prevIdxOffset * S + i - 2]; // alphas_a[prevIdxOffset][i -
                                                  // 2];
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
      alphas_a[curIdxOffset * S + i] = result +
          logProbs_a[batchIndex][t][labelIdx]; // alphas_a[curIdxOffset][i]
    }
  }
  auto idx1 = (T - 1) % 2;
  auto ltrIdx = alphas_a[S * idx1 + S - 1] > alphas_a[S * idx1 + S - 2]
      ? S - 1
      : S - 2; // alphas_a[idx1][S - 1], alphas_a[idx1][S - 2]
  delete[] alphas_a;
  // path stores the token index for each time step after force alignment.
  for (auto t = T - 1; t > -1; t--) {
    auto lbl_idx = ltrIdx % 2 == 0 ? blank : targets_a[batchIndex][ltrIdx / 2];
    paths_a[batchIndex][t] = lbl_idx;
    ltrIdx -= backPtr_a[t * S + ltrIdx]; // backPtr_a[t][ltrIdx]
  }
  delete[] backPtr_a;
}

template <typename scalar_t>
const auto forced_align_long_impl =
    forced_align_impl<scalar_t, ScalarType::Long>;

template <typename scalar_t>
const auto forced_align_int_impl = forced_align_impl<scalar_t, ScalarType::Int>;

std::tuple<Tensor, Tensor> compute(
    Tensor logProbs,
    Tensor targets,
    Tensor inputLengths,
    Tensor targetLengths,
    const int64_t blank) {
  STD_TORCH_CHECK(logProbs.is_cpu(), "log_probs must be a CPU tensor");
  STD_TORCH_CHECK(targets.is_cpu(), "targets must be a CPU tensor");
  STD_TORCH_CHECK(inputLengths.is_cpu(), "input_lengths must be a CPU tensor");
  STD_TORCH_CHECK(
      targetLengths.is_cpu(), "target_lengths must be a CPU tensor");
  STD_TORCH_CHECK(
      logProbs.scalar_type() == ScalarType::Double ||
          logProbs.scalar_type() == ScalarType::Float ||
          logProbs.scalar_type() == ScalarType::Half,
      "log_probs must be float64, float32 or float16 (half) type");
  STD_TORCH_CHECK(
      targets.scalar_type() == ScalarType::Int ||
          targets.scalar_type() == ScalarType::Long,
      "targets must be int32 or int64 type");
  STD_TORCH_CHECK(logProbs.is_contiguous(), "log_probs must be contiguous");
  STD_TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  STD_TORCH_CHECK(
      logProbs.dim() == 3,
      "log_probs must be 3-D (batch_size, input length, num classes)");
  STD_TORCH_CHECK(
      targets.dim() == 2, "targets must be 2-D (batch_size, target length,)");
  STD_TORCH_CHECK(
      inputLengths.dim() == 1, "input_lengths must be 1-D (batch_size,)");
  STD_TORCH_CHECK(
      targetLengths.dim() == 1, "target_lengths must be 1-D (batch_size,)");
  STD_TORCH_CHECK(
      logProbs.size(0) == 1,
      "The batch dimension for log_probs must be 1 at the current version.")
  STD_TORCH_CHECK(
      targets.size(0) == 1,
      "The batch dimension for targets must be 1 at the current version.")
  STD_TORCH_CHECK(
      blank >= 0 && blank < logProbs.size(-1),
      "blank must be within [0, num classes)");
  THO_DISPATCH_V2(
      inputLengths.scalar_type(),
      "forced_align_impl",
      AT_WRAP([&] {
        STD_TORCH_CHECK(
            logProbs.size(1) == torchaudio::util::max<scalar_t>(inputLengths),
            "input length mismatch");
      }),
      ScalarType::Int,
      ScalarType::Long);
  THO_DISPATCH_V2(
      targetLengths.scalar_type(),
      "forced_align_impl",
      AT_WRAP([&] {
        STD_TORCH_CHECK(
            targets.size(1) == torchaudio::util::max<scalar_t>(targetLengths),
            "target length mismatch");
      }),
      ScalarType::Int,
      ScalarType::Long);
  const auto B = logProbs.size(0);
  const auto T = logProbs.size(1);
  Tensor paths = torchaudio::stable::new_zeros(targets, {B, T});
  THO_DISPATCH_V2(
      logProbs.scalar_type(),
      "forced_align_impl",
      AT_WRAP([&] {
        if (targets.scalar_type() == ScalarType::Long) {
          forced_align_long_impl<scalar_t>(logProbs, targets, blank, paths);
        } else {
          forced_align_int_impl<scalar_t>(logProbs, targets, blank, paths);
        }
      }),
      AT_EXPAND(AT_FLOATING_TYPES),
      ScalarType::Half);
  return std::make_tuple(paths, logProbs);
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("forced_align", TORCH_BOX(&compute));
}

} // namespace cpu
} // namespace alignment
} // namespace torchaudio
