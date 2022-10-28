import torch
import torchaudio.prototype.functional as F
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin, torch_script


class TorchScriptConsistencyTestImpl(TestBaseMixin):
    def _assert_consistency(self, func, inputs, shape_only=False):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(device=self.device, dtype=self.dtype)
            inputs_.append(i)
        ts_func = torch_script(func)

        torch.random.manual_seed(40)
        output = func(*inputs_)

        torch.random.manual_seed(40)
        ts_output = ts_func(*inputs_)

        if shape_only:
            ts_output = ts_output.shape
            output = output.shape
        self.assertEqual(ts_output, output)

    @nested_params(
        [F.convolve, F.fftconvolve],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (2, 3, 2)
        L_x, L_y = 32, 55
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        self._assert_consistency(fn, (x, y, mode))

    def test_add_noise(self):
        leading_dims = (2, 3)
        L = 31

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        self._assert_consistency(F.add_noise, (waveform, noise, lengths, snr))
