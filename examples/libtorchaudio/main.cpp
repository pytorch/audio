#include <torch/script.h>

int main(int argc, char* argv[]) {
  if (argc !=4) {
    std::cerr << "Usage: " << argv[0] << " <JIT_OBJECT> <INPUT_FILE> <OUTPUT_FILE>" << std::endl;
    return -1;
  }

  torch::jit::script::Module module;
  std::cout << "Loading module from: " << argv[0] << std::endl;
  try {
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &error) {
    std::cerr << "Failed to load the module:" << error.what() << std::endl;
    return -1;
  }

  std::cout << "Performing the process ..." << std::endl;
  module.forward({c10::IValue(argv[2]), c10::IValue(argv[3])});
  std::cout << "Done." << std::endl; 
}
