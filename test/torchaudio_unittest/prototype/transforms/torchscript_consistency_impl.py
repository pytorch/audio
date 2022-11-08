import torch
import torchaudio.prototype.transforms as T
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin, torch_script


class Transforms(TestBaseMixin):
    @nested_params(
        [T.Convolve, T.FFTConvolve],
        ["full", "valid", "same"],
    )
    def test_Convolve(self, cls, mode):
        leading_dims = (2, 3, 2)
        L_x, L_y = 32, 55
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        convolve = cls(mode=mode).to(device=self.device, dtype=self.dtype)
        output = convolve(x, y)
        ts_output = torch_script(convolve)(x, y)
        self.assertEqual(ts_output, output)
