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
