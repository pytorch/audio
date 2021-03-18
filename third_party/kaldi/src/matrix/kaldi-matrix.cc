#include "matrix/kaldi-matrix.h"
#include <torch/torch.h>

namespace {

template <typename Real>
void assert_matrix_shape(const torch::Tensor& tensor_);

template <>
void assert_matrix_shape<float>(const torch::Tensor& tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 2);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat32);
  TORCH_CHECK(tensor_.device().is_cpu(), "Input tensor has to be on CPU.");
}

template <>
void assert_matrix_shape<double>(const torch::Tensor& tensor_) {
  TORCH_INTERNAL_ASSERT(tensor_.ndimension() == 2);
  TORCH_INTERNAL_ASSERT(tensor_.dtype() == torch::kFloat64);
  TORCH_CHECK(tensor_.device().is_cpu(), "Input tensor has to be on CPU.");
}

} // namespace

namespace kaldi {

template <typename Real>
MatrixBase<Real>::MatrixBase(torch::Tensor tensor) : tensor_(tensor) {
  assert_matrix_shape<Real>(tensor_);
};

template class Matrix<float>;
template class Matrix<double>;
template class MatrixBase<float>;
template class MatrixBase<double>;
template class SubMatrix<float>;
template class SubMatrix<double>;

} // namespace kaldi
