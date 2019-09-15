#include <string>

namespace at {
struct Tensor;
} // namespace at

namespace torch { namespace audio {

/// Takes a tensor and performs a biquad difference function

int biquad(
    at::Tensor& input_waveform,
    at::Tensor output_waveform,
    float a0,
    float a1,
    float a2,
    float b0,
    float b1,
    float b2
);
}} // namespace torch::audio
