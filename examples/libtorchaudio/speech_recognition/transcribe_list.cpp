#include <chrono>
#include <torch/script.h>


int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << "<JIT_OBJECT_DIR> <FILE_LIST> <OUTPUT_DIR>\n" << std::endl;
    std::cerr << "<FILE_LIST> is `<ID>\t<PATH>\t<TRANSCRIPTION>`" << std::endl;
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

  std::ifstream input_file(argv[2]);
  std::string output_dir(argv[3]);
  std::ofstream output_ref(output_dir + "/ref.trn");
  std::ofstream output_hyp(output_dir + "/hyp.trn");
  std::string line;
  std::chrono::milliseconds t_encode(0);
  std::chrono::milliseconds t_decode(0);
  while(std::getline(input_file, line)) {
    std::istringstream iline(line);
    std::string id;
    std::string path;
    std::string reference;
    std::getline(iline, id, '\t');
    std::getline(iline, path, '\t');
    std::getline(iline, reference, '\t');

    auto waveform = loader.forward({c10::IValue(path)});
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    auto emission = encoder.forward({waveform});
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto result = decoder.forward({emission});
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    t_encode += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    t_decode += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    auto hypothesis = result.toString()->string();
    output_hyp << hypothesis << " (" << id << ")" << std::endl;
    output_ref << reference << " (" << id << ")" << std::endl;
    std::cout << id << '\t' << hypothesis << std::endl;
  }
  std::cout << "Time (encode): " << t_encode.count() << " [ms]" << std::endl;
  std::cout << "Time (decode): " << t_decode.count() << " [ms]" << std::endl;
}
