import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


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

    @parameterized.expand(
        [
            (F.convolve,),
            (F.fftconvolve,),
        ]
    )
    def test_convolve(self, fn):
        leading_dims = (2, 3, 2)
        L_x, L_y = 32, 55
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        self._assert_consistency(fn, (x, y))

    def test_add_noise(self):
        leading_dims = (2, 3)
        L = 31

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        self._assert_consistency(F.add_noise, (waveform, noise, lengths, snr))

    def test_simulate_rir_ism_single_band(self):
        room_dim = torch.tensor([9.0, 9.0, 9.0], dtype=self.dtype, device=self.device)
        mic_array = torch.tensor([1, 1, 1], dtype=self.dtype, device=self.device).reshape(1, -1).repeat(6, 1)
        source = torch.tensor([7, 7, 7], dtype=self.dtype, device=self.device)
        max_order = 3
        e_absorption = 0.5
        center_frequency = torch.tensor([125, 250, 500, 1000, 2000, 4000, 8000], dtype=self.dtype, device=self.device)
        self._assert_consistency(
            F.simulate_rir_ism,
            (room_dim, source, mic_array, max_order, e_absorption, 1000, 81, center_frequency, 343.0, 16000.0),
        )

    def test_simulate_rir_ism_multi_band(self):
        room_dim = torch.tensor([9.0, 9.0, 9.0], dtype=self.dtype, device=self.device)
        mic_array = torch.tensor([1, 1, 1], dtype=self.dtype, device=self.device).reshape(1, -1).repeat(6, 1)
        source = torch.tensor([7, 7, 7], dtype=self.dtype, device=self.device)
        max_order = 3
        e_absorption = torch.rand(7, 6, dtype=self.dtype, device=self.device)
        center_frequency = torch.tensor([125, 250, 500, 1000, 2000, 4000, 8000], dtype=self.dtype, device=self.device)
        self._assert_consistency(
            F.simulate_rir_ism,
            (room_dim, source, mic_array, max_order, e_absorption, 1000, 81, center_frequency, 343.0, 16000.0),
        )
