#include <torch/script.h>

namespace {

int64_t cpu_lfilter_core_loop(
  const torch::Tensor& input_signal_windows,
  const torch::Tensor& a_coeff_flipped,
  torch::Tensor& padded_output_waveform) {
   
    TORCH_CHECK(input_signal_windows.device().is_cpu() &&
                a_coeff_flipped.device().is_cpu() &&
                padded_output_waveform.device().is_cpu());

    TORCH_CHECK(input_signal_windows.is_contiguous() && 
                a_coeff_flipped.is_contiguous() &&
                padded_output_waveform.is_contiguous());

    TORCH_CHECK(input_signal_windows.dtype().name()=="float" &&
                a_coeff_flipped.dtype().name()=="float" &&
                padded_output_waveform.dtype().name()=="float");
                
    TORCH_CHECK(input_signal_windows.size(0)==padded_output_waveform.size(0));

    TORCH_CHECK(input_signal_windows.size(1)+a_coeff_flipped.size(0)-1==padded_output_waveform.size(1));
   
    int64_t n_channel = input_signal_windows.size(0);
    int64_t n_samples_input = input_signal_windows.size(1);
    int64_t n_samples_output = padded_output_waveform.size(1);
    int64_t n_order = a_coeff_flipped.size(0);
    float* output_data = padded_output_waveform.data_ptr<float>();
    const float* input_data = input_signal_windows.data_ptr<float>();
    const float* a_coeff_flipped_data = a_coeff_flipped.data_ptr<float>();
    for(int64_t i_channel = 0; i_channel<n_channel; i_channel++){
      for(int64_t i_sample = 0; i_sample<n_samples_input; i_sample++){
        int64_t offset_input = i_channel*n_samples_input;
        int64_t offset_output = i_channel*n_samples_output;
        float a0 = input_data[offset_input+i_sample]; 
        for(int64_t i_coeff = 0; i_coeff<n_order; i_coeff++){
          a0-=output_data[offset_output+i_sample+i_coeff]*a_coeff_flipped_data[i_coeff];
        }
        output_data[offset_output+i_sample+n_order-1] = a0;
      }
    }
  return 0;
}

} // namespace

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("_lfilter_core_loop", &cpu_lfilter_core_loop);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
       "_lfilter_core_loop(Tensor input_signal_windows,"
       "Tensor a_coeff_flipped_flipped,"
       "Tensor padded_output_waveform) -> int");
}