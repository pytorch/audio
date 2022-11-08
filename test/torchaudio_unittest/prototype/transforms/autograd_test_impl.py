from typing import List

import torch
import torchaudio.prototype.transforms as T
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin


class Autograd(TestBaseMixin):
    def assert_grad(
        self,
        transform: torch.nn.Module,
        inputs: List[torch.Tensor],
        *,
        nondet_tol: float = 0.0,
    ):
        transform = transform.to(dtype=torch.float64, device=self.device)

        # gradcheck and gradgradcheck only pass if the input tensors are of dtype `torch.double` or
        # `torch.cdouble`, when the default eps and tolerance values are used.
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(dtype=torch.cdouble if i.is_complex() else torch.double, device=self.device)
                i.requires_grad = True
            inputs_.append(i)
        assert gradcheck(transform, inputs_)
        assert gradgradcheck(transform, inputs_, nondet_tol=nondet_tol)

    @nested_params(
        [T.Convolve, T.FFTConvolve],
        ["full", "valid", "same"],
    )
    def test_Convolve(self, cls, mode):
        leading_dims = (4, 3, 2)
        L_x, L_y = 23, 40
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)
        convolve = cls(mode=mode).to(dtype=self.dtype, device=self.device)
        self.assert_grad(convolve, [x, y])
