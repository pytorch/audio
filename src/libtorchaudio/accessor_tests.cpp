#include <libtorchaudio/accessor.h>
#include <cstdint>
#include <torch/torch.h>

using namespace std;

bool test_accessor(const torch::Tensor& tensor) {
  int64_t* data_ptr = tensor.template data_ptr<int64_t>();
  auto accessor = Accessor<3, int64_t>(tensor);
  for (int i = 0; i < tensor.size(0); i++) {
    for (int j = 0; j < tensor.size(1); j++) {
      for (int k = 0; k < tensor.size(2); k++) {
        auto check = *(data_ptr++) ==  accessor.index(i, j, k);
        if (!check) {
          return false;
        }
      }
    }
  }
  return true;
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("torchaudio::_test_accessor", &test_accessor);
}
