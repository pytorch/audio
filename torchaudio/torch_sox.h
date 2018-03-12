#include <string>
#include <tuple>

namespace at {
struct Tensor;
} // namespace at

namespace torch { namespace audio {

/// Reads an audio file from the given `path` into the `output` `Tensor` and
/// returns the sample rate of the audio file.
/// Throws `std::runtime_error` if the audio file could not be opened, or an
/// error ocurred during reading of the audio data.
int read_audio_file(const std::string& path, at::Tensor output);

/// Writes the data of a `Tensor` into an audio file at the given `path`, with
/// a certain extension (e.g. `wav`or `mp3`) and sample rate.
/// Throws `std::runtime_error` when the audio file could not be opened for
/// writing, or an error ocurred during writing of the audio data.
void write_audio_file(
    const std::string& path,
    at::Tensor tensor,
    const std::string& extension,
    int sample_rate);
}} // namespace torch::audio
