import torch
import torchaudio.functional as F
from torch.autograd import gradcheck
from torchaudio_unittest import common_utils


class Autograd(common_utils.TestBaseMixin):
    def test_x_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        x.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_a_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        a.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_b_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        b.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_all_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        b.requires_grad = True
        a.requires_grad = True
        x.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)
