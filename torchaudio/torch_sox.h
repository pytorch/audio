#include <string>

namespace at {
struct Tensor;
} // namespace at

namespace torch { namespace audio {

/// Reads an audio file from the given `path` into the `output` `Tensor` and
/// returns the sample rate of the audio file.
/// Throws `std::runtime_error` if the audio file could not be opened, or an
/// error ocurred during reading of the audio data.
int read_audio_file(
    const std::string& path,
    at::Tensor output,
    int64_t number_of_samples,
    int64_t offset);

/// Writes the data of a `Tensor` into an audio file at the given `path`, with
/// a certain extension (e.g. `wav`or `mp3`) and sample rate.
/// Throws `std::runtime_error` when the audio file could not be opened for
/// writing, or an error ocurred during writing of the audio data.
void write_audio_file(
    const std::string& path,
    at::Tensor tensor,
    const std::string& extension,
    int sample_rate,
    int precision);

 /// Reads an audio file from the given `path` and returns a tuple of
/// the number of channels, length in samples, sample rate, and bits / sec.
/// Throws `std::runtime_error` if the audio file could not be opened, or an
/// error ocurred during reading of the audio data.
std::tuple<int64_t, int64_t, int64_t, int64_t> get_info(
    const std::string& file_name);
}} // namespace torch::audio
