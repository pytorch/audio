#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <torch/script.h>
#include "rnnt.h"

namespace {

int64_t cpu_rnnt_loss(
    torch::Tensor acts,
    torch::Tensor labels,
    torch::Tensor input_lengths,
    torch::Tensor label_lengths,
    torch::Tensor costs,
    torch::Tensor grads,
    int64_t blank_label,
    int64_t num_threads) {
  TORCH_CHECK(labels.dtype() == torch::kInt32, "labels must be int32 type");
  TORCH_CHECK(
      label_lengths.dtype() == torch::kInt32,
      "label_lengths must be int32 type");
  TORCH_CHECK(
      input_lengths.dtype() == torch::kInt32, "lengths must be int32 type");
  TORCH_CHECK(acts.is_contiguous(), "acts must be contiguous");
  TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
  TORCH_CHECK(
      label_lengths.is_contiguous(), "label_lengths must be contiguous");
  TORCH_CHECK(input_lengths.is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(
      input_lengths.size(0) == acts.size(0),
      "batch dimension mismatch between acts and input_lengths: each example must have a length");
  TORCH_CHECK(
      label_lengths.size(0) == acts.size(0),
      "batch dimension mismatch between acts and label_lengths: each example must have a label length");
  TORCH_CHECK(acts.dim() == 4, "acts must be 4-D (batch, time, label, class)");
  TORCH_CHECK(
      labels.dim() == 2, "labels must be 2-D (batch, max label length)");
  TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1-D");
  TORCH_CHECK(label_lengths.dim() == 1, "label_lengths must be 1-D");

  int maxT = acts.size(1);
  int maxU = acts.size(2);
  int minibatch_size = acts.size(0);
  int alphabet_size = acts.size(3);

  TORCH_CHECK(
      at::max(input_lengths).item().toInt() == maxT, "input length mismatch");
  TORCH_CHECK(
      at::max(label_lengths).item().toInt() + 1 == maxU,
      "output length mismatch");

  rnntOptions options;
  memset(&options, 0, sizeof(options));
  options.maxT = maxT;
  options.maxU = maxU;
  options.blank_label = blank_label;
  options.batch_first = true;
  options.loc = RNNT_CPU;
  options.num_threads = num_threads;

  // have to use at least one
  options.num_threads = std::max(options.num_threads, (unsigned int)1);

  size_t cpu_size_bytes = 0;
  switch (acts.scalar_type()) {
    case torch::ScalarType::Float: {
      get_workspace_size(maxT, maxU, minibatch_size, false, &cpu_size_bytes);

      std::vector<float> cpu_workspace(cpu_size_bytes / sizeof(float), 0);

      compute_rnnt_loss(
          acts.data_ptr<float>(),
          grads.data_ptr<float>(),
          labels.data_ptr<int>(),
          label_lengths.data_ptr<int>(),
          input_lengths.data_ptr<int>(),
          alphabet_size,
          minibatch_size,
          costs.data_ptr<float>(),
          cpu_workspace.data(),
          options);

      return 0;
    }
    case torch::ScalarType::Double: {
      get_workspace_size(
          maxT, maxU, minibatch_size, false, &cpu_size_bytes, sizeof(double));

      std::vector<double> cpu_workspace(cpu_size_bytes / sizeof(double), 0);

      compute_rnnt_loss_fp64(
          acts.data_ptr<double>(),
          grads.data_ptr<double>(),
          labels.data_ptr<int>(),
          label_lengths.data_ptr<int>(),
          input_lengths.data_ptr<int>(),
          alphabet_size,
          minibatch_size,
          costs.data_ptr<double>(),
          cpu_workspace.data(),
          options);

      return 0;
    }
    default:
      TORCH_CHECK(
          false,
          std::string(__func__) + " not implemented for '" +
              toString(acts.scalar_type()) + "'");
  }
  return -1;
}

} // namespace

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("rnnt_loss", &cpu_rnnt_loss);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "rnnt_loss(Tensor acts,"
      "Tensor labels,"
      "Tensor input_lengths,"
      "Tensor label_lengths,"
      "Tensor costs,"
      "Tensor grads,"
      "int blank_label,"
      "int num_threads) -> int");
}
