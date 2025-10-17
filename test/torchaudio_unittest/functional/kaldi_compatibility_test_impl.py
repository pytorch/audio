import torch
import torchaudio.functional as F
from torchaudio_unittest.common_utils import RequestMixin, TempDirMixin, TestBaseMixin
from torchaudio_unittest.common_utils.kaldi_utils import convert_args, run_kaldi


class Kaldi(TempDirMixin, TestBaseMixin, RequestMixin):
    def assert_equal(self, output, *, expected, rtol=None, atol=None):
        expected = expected.to(dtype=self.dtype, device=self.device)
        self.assertEqual(output, expected, rtol=rtol, atol=atol)

    def test_sliding_window_cmn(self):
        """sliding_window_cmn should be numerically compatible with apply-cmvn-sliding"""
        kwargs = {
            "cmn_window": 600,
            "min_cmn_window": 100,
            "center": False,
            "norm_vars": False,
        }

        torch.manual_seed(0)
        tensor = torch.randn(40, 10, dtype=self.dtype).to(self.device)
        result = F.sliding_window_cmn(tensor, **kwargs)
        command = ["apply-cmvn-sliding"] + convert_args(**kwargs) + ["ark:-", "ark:-"]
        kaldi_result = run_kaldi(self.request, command, "ark", tensor)
        self.assert_equal(result, expected=kaldi_result)
