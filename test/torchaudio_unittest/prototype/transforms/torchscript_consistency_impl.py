import torch
import torchaudio.prototype.transforms as T
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin, torch_script


class Transforms(TestBaseMixin):
    @nested_params(
        ["Convolve", "FFTConvolve"],
        ["full", "valid", "same"],
    )
    def test_Convolve(self, cls, mode):
        leading_dims = (2, 3, 2)
        L_x, L_y = 32, 55
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        convolve = getattr(T, cls)(mode=mode).to(device=self.device, dtype=self.dtype)
        output = convolve(x, y)
        ts_output = torch_script(convolve)(x, y)
        self.assertEqual(ts_output, output)

    def test_Speed(self):
        leading_dims = (3, 2)
        time = 200
        waveform = torch.rand(*leading_dims, time, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.randint(1, time, leading_dims, dtype=self.dtype, device=self.device)

        speed = T.Speed(1000, 0.9).to(self.device, self.dtype)
        output = speed(waveform, lengths)
        ts_output = torch_script(speed)(waveform, lengths)
        self.assertEqual(ts_output, output)

    def test_SpeedPerturbation(self):
        leading_dims = (3, 2)
        time = 200
        waveform = torch.rand(*leading_dims, time, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.randint(1, time, leading_dims, dtype=self.dtype, device=self.device)

        speed = T.SpeedPerturbation(1000, [0.9]).to(self.device, self.dtype)
        output = speed(waveform, lengths)
        ts_output = torch_script(speed)(waveform, lengths)
        self.assertEqual(ts_output, output)

    def test_AddNoise(self):
        leading_dims = (2, 3)
        L = 31

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        add_noise = T.AddNoise().to(self.device, self.dtype)
        output = add_noise(waveform, noise, lengths, snr)
        ts_output = torch_script(add_noise)(waveform, noise, lengths, snr)
        self.assertEqual(ts_output, output)

    def test_Preemphasis(self):
        waveform = torch.rand(3, 4, 10, dtype=self.dtype, device=self.device)
        preemphasis = T.Preemphasis(coeff=0.97).to(dtype=self.dtype, device=self.device)
        output = preemphasis(waveform)
        ts_output = torch_script(preemphasis)(waveform)
        self.assertEqual(ts_output, output)

    def test_Deemphasis(self):
        waveform = torch.rand(3, 4, 10, dtype=self.dtype, device=self.device)
        deemphasis = T.Deemphasis(coeff=0.97).to(dtype=self.dtype, device=self.device)
        output = deemphasis(waveform)
        ts_output = torch_script(deemphasis)(waveform)
        self.assertEqual(ts_output, output)
