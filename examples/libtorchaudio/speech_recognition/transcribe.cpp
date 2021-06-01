#include <torch/script.h>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <JIT_OBJECT_DIR> <INPUT_AUDIO_FILE>" << std::endl;
    return -1;
  }

  torch::jit::script::Module loader, encoder, decoder;
  std::cout << "Loading module from: " << argv[1] << std::endl;
  try {
    loader = torch::jit::load(std::string(argv[1]) + "/loader.zip");
  } catch (const c10::Error &error) {
    std::cerr << "Failed to load the module:" << error.what() << std::endl;
    return -1;
  }
  try {
    encoder = torch::jit::load(std::string(argv[1]) + "/encoder.zip");
  } catch (const c10::Error &error) {
    std::cerr << "Failed to load the module:" << error.what() << std::endl;
    return -1;
  }
  try {
    decoder = torch::jit::load(std::string(argv[1]) + "/decoder.zip");
  } catch (const c10::Error &error) {
    std::cerr << "Failed to load the module:" << error.what() << std::endl;
    return -1;
  }

  std::cout << "Loading the audio" << std::endl;
  auto waveform = loader.forward({c10::IValue(argv[2])});
  std::cout << "Running inference" << std::endl;
  auto emission = encoder.forward({waveform});
  std::cout << "Generating the transcription" << std::endl;
  auto result = decoder.forward({emission});
  std::cout << result.toString()->string() << std::endl;
  std::cout << "Done." << std::endl; 
}
