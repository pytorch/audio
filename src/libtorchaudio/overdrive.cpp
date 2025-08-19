#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <libtorchaudio/accessor.h>

using namespace std;

namespace torchaudio {

using torch::stable::Tensor;

template <typename scalar_t>
void overdrive_cpu_kernel(
    Accessor<2, scalar_t> waveform_accessor,
    Accessor<2, scalar_t> temp_accessor,
    Accessor<1, scalar_t, false> last_in_accessor,
    Accessor<1, scalar_t> last_out_accessor,
    Accessor<2, scalar_t, false> output_waveform_accessor) {
  int64_t n_frames = waveform_accessor.size(1);
  int64_t n_channels = waveform_accessor.size(0);

  at::parallel_for(0, n_channels, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i_channel = begin; i_channel < end; ++i_channel) {
      for (int64_t i_frame = 0; i_frame < n_frames; ++i_frame) {
        last_out_accessor.set_index(
          temp_accessor.index(i_channel, i_frame) -
            last_in_accessor.index(i_channel) + 0.995 * last_out_accessor.index(i_channel),
          i_channel);
        last_in_accessor.set_index(temp_accessor.index(i_channel, i_frame), i_channel);
        output_waveform_accessor.set_index(
          waveform_accessor.index(i_channel, i_frame) * 0.5 +
            last_out_accessor.index(i_channel) * 0.75,
          i_channel, i_frame);
      }
    }
  });
}

void overdrive_core_loop_cpu(
  const Tensor waveform,
  const Tensor temp,
  Tensor last_in,
  const Tensor last_out,
  Tensor output_waveform) {
    int32_t dtype;
    aoti_torch_get_dtype(waveform.get(), &dtype);
    if (dtype == aoti_torch_dtype_float64()) {
      overdrive_cpu_kernel<double>(
        Accessor<2, double>(wave_acc),
        Accessor<2, double>(temp_acc),
        Accessor<1, double>(last_in_acc),
        Accessor<1, double>(last_out_acc),
        Accessor<2, double>(out_acc));
    } else if (dtype == aoti_torch_dtype_float32()) {
      overdrive_cpu_kernel<float>(
        Accessor<2, float>(wave_acc),
        Accessor<2, float>(temp_acc),
        Accessor<1, float>(last_in_acc),
        Accessor<1, float>(last_out_acc),
        Accessor<2, float>(out_acc));
    } else if (dtype == aoti_torch_dtype_float16()) {
      overdrive_cpu_kernel<c10::Half>(
        Accessor<2, c10::Half>(wave_acc),
        Accessor<2, c10::Half>(temp_acc),
        Accessor<1, c10::Half>(last_in_acc),
        Accessor<1, c10::Half>(last_out_acc),
        Accessor<2, c10::Half>(out_acc));
    }
}



void boxed_overdrive_core_loop(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor t1(to<AtenTensorHandle>(stack[0]));
  Tensor t2(to<AtenTensorHandle>(stack[1]));
  Tensor t3(to<AtenTensorHandle>(stack[2]));
  Tensor t4(to<AtenTensorHandle>(stack[3]));
  Tensor t5(to<AtenTensorHandle>(stack[4]));
  overdrive_core_loop(
      std::move(t1), std::move(t2), std::move(t3), std::move(t4), std::move(t5));
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
    "overdrive_core_loop(Tensor waveform,"
    "Tensor temp, Tensor last_in, Tensor last_out,"
    "Tensor output_waveform)"
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("overdrive_core_loop", &overdrive_core_loop_cpu);
}

} // namespace
