#include <libtorchaudio/accessor.h>
#include <cstdint>
#include <torch/csrc/stable/library.h>

namespace torchaudio {

namespace accessor_tests {

using namespace std;
using torch::stable::Tensor;

bool test_accessor(const Tensor tensor) {
  int64_t* data_ptr = (int64_t*)tensor.data_ptr();
  auto accessor = Accessor<3, int64_t>(tensor);
  for (unsigned int i = 0; i < tensor.size(0); i++) {
    for (unsigned int j = 0; j < tensor.size(1); j++) {
      for (unsigned int k = 0; k < tensor.size(2); k++) {
        auto check = *(data_ptr++) ==  accessor.index(i, j, k);
        if (!check) {
          return false;
        }
      }
    }
  }
  return true;
}

void boxed_test_accessor(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
  Tensor t1(to<AtenTensorHandle>(stack[0]));
  auto result = test_accessor(std::move(t1));
  stack[0] = from(result);
}

STABLE_TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "_test_accessor(Tensor log_probs) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("torchaudio::_test_accessor", &boxed_test_accessor);
}

}
}
