import torch
import torchaudio.functional as F
from torchaudio_unittest import common_utils


class NumericalStability(common_utils.TestBaseMixin):
    def test_9th_order_filter(self):
        """
        the filter coefficients is get by `scipy.signal.butter(9, 850, 'hp', fs=22050, output='ba')`
        """
        x = torch.zeros(1024, dtype=self.dtype, device=self.device)
        x[0] = 1
        a = torch.tensor([1., -7.60545606, 25.80187885, -51.23717435,
                          65.62093428, -56.20096964, 32.18274279, -11.88025302,
                          2.56506938, -0.24677075], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.49676025, -4.47084227, 17.88336908, -41.72786118,
                          62.59179178, -62.59179178, 41.72786118, -17.88336908,
                          4.47084227, -0.49676025], dtype=self.dtype, device=self.device)
        y = F.lfilter(x, a, b, False)
        if self.dtype == torch.float64:
            assert torch.all(y.abs() < 0.8)
        else:
            assert torch.any(y.abs() > 1)
