#include "matrix/kaldi-vector.h"
#include <torch/torch.h>
#include "matrix/kaldi-matrix.h"

namespace {

template <typename Real>
void assert_vector_shape(const torch::Tensor& tensor_);

template <>
void assert_vector_shape<float>(const torch::Tensor& tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 1);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat32);
  TORCH_CHECK(tensor_.device().is_cpu(), "Input tensor has to be on CPU.");
}

template <>
void assert_vector_shape<double>(const torch::Tensor& tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 1);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat64);
  TORCH_CHECK(tensor_.device().is_cpu(), "Input tensor has to be on CPU.");
}

} // namespace

namespace kaldi {

template <typename Real>
VectorBase<Real>::VectorBase(torch::Tensor tensor)
    : tensor_(tensor), data_(tensor.data_ptr<Real>()) {
  assert_vector_shape<Real>(tensor_);
};

template <typename Real>
VectorBase<Real>::VectorBase() : VectorBase<Real>(torch::empty({0})) {}

template class Vector<float>;
template class Vector<double>;
template class VectorBase<float>;
template class VectorBase<double>;

} // namespace kaldi
