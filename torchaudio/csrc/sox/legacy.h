#include <sox.h>
#include <torch/torch.h>

namespace torch { namespace audio {

/// Reads an audio file from the given `path` into the `output` `Tensor` and
/// returns the sample rate of the audio file.
/// Throws `std::runtime_error` if the audio file could not be opened, or an
/// error occurred during reading of the audio data.
int read_audio_file(
    const std::string& file_name,
    at::Tensor output,
    bool ch_first,
    int64_t nframes,
    int64_t offset,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* ft);

/// Writes the data of a `Tensor` into an audio file at the given `path`, with
/// a certain extension (e.g. `wav`or `mp3`) and sample rate.
/// Throws `std::runtime_error` when the audio file could not be opened for
/// writing, or an error occurred during writing of the audio data.
void write_audio_file(
    const std::string& file_name,
    const at::Tensor& tensor,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* file_type);

/// Reads an audio file from the given `path` and returns a tuple of
/// sox_signalinfo_t and sox_encodinginfo_t, which contain information about
/// the audio file such as sample rate, length, bit precision, encoding and more.
/// Throws `std::runtime_error` if the audio file could not be opened, or an
/// error occurred during reading of the audio data.
std::tuple<sox_signalinfo_t, sox_encodinginfo_t> get_info(
    const std::string& file_name);
}} // namespace torch::audio
