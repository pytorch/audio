import torch
from torch.autograd import gradcheck, gradgradcheck
import torchaudio.prototype.functional as F
from torchaudio_unittest.common_utils import TestBaseMixin


class AutogradTestImpl(TestBaseMixin):
    def test_convolve(self):
        leading_dims = (4, 3, 2)
        L_x, L_y = 23, 40
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device, requires_grad=True)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device, requires_grad=True)
        self.assertTrue(gradcheck(F.convolve, (x, y)))
        self.assertTrue(gradgradcheck(F.convolve, (x, y)))
