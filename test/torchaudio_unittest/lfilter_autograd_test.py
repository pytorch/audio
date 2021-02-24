import torch
import torchaudio.functional as F
from parameterized import parameterized
from torch.autograd import gradcheck
from torchaudio_unittest import common_utils



class AutogradLfilter(common_utils.TestBaseMixin):
    def _get_a(self):
        return torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)

    def _get_b(self):
        return torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)

    def _get_waveform(self):
        torch.random.manual_seed(2434)
        return torch.rand(2, 256 * 2, dtype=self.dtype, device=self.device)

    def test_x_grad(self):
        x, a, b = self._get_waveform(), self._get_a(), self._get_b()
        x.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_a_grad(self):
        x, a, b = self._get_waveform(), self._get_a(), self._get_b()
        a.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_b_grad(self):
        x, a, b = self._get_waveform(), self._get_a(), self._get_b()
        b.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)
    
    def test_all_grad(self):
        x, a, b = self._get_waveform(), self._get_a(), self._get_b()
        b.requires_grad = True
        a.requires_grad = True
        x.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

class TestAutogradLfilterCPU(AutogradLfilter, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')

@common_utils.skipIfNoCuda
class TestAutogradLfilterCUDA(AutogradLfilter, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cuda')