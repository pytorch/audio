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
    const std::string& file_name,
    at::Tensor output,
    bool ch_first,
    int64_t nframes,
    int64_t offset,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* ft)

/// Writes the data of a `Tensor` into an audio file at the given `path`, with
/// a certain extension (e.g. `wav`or `mp3`) and sample rate.
/// Throws `std::runtime_error` when the audio file could not be opened for
/// writing, or an error ocurred during writing of the audio data.
void write_audio_file(
    const std::string& file_name,
    at::Tensor& tensor,
    sox_signalinfo_t* si,
    sox_encodinginfo_t* ei,
    const char* file_type)

/// Reads an audio file from the given `path` and returns a tuple of
/// sox_signalinfo_t and sox_encodinginfo_t, which contain information about
/// the audio file such as sample rate, length, bit precision, encoding and more.
/// Throws `std::runtime_error` if the audio file could not be opened, or an
/// error ocurred during reading of the audio data.
std::tuple<sox_signalinfo_t, sox_encodinginfo_t> get_info(
    const std::string& file_name);

// get names of all sox effects
std::vector<std::string> get_effect_names();

// Initialize and Shutdown SoX effects chain.  These functions should only be run once.
int initialize_sox();
int shutdown_sox();

// Struct for build_flow_effects function
struct SoxEffect {
  SoxEffect() : ename(""), eopts({""})  { }
  std::string ename;
  std::vector<std::string> eopts;
};

/// Build a SoX chain, flow the effects, and capture the results in a tensor.
/// An audio file from the given `path` flows through an effects chain given
/// by a list of effects and effect options to an output buffer which is encoded
/// into memory to a target signal type and target signal encoding.  The resulting
/// buffer is then placed into a tensor.  This function returns the output tensor
/// and the sample rate of the output tensor.
int build_flow_effects(const std::string& file_name,
                       at::Tensor otensor,
                       bool ch_first,
                       sox_signalinfo_t* target_signal,
                       sox_encodinginfo_t* target_encoding,
                       const char* file_type,
                       std::vector<SoxEffect> pyeffs,
                       int max_num_eopts);
}} // namespace torch::audio
