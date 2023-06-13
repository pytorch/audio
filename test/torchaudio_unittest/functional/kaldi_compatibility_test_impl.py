import torch
import torchaudio.functional as F
from torchaudio_unittest.common_utils import skipIfNoExec, TempDirMixin, TestBaseMixin
from torchaudio_unittest.common_utils.kaldi_utils import convert_args, run_kaldi


class Kaldi(TempDirMixin, TestBaseMixin):
    def assert_equal(self, output, *, expected, rtol=None, atol=None):
        expected = expected.to(dtype=self.dtype, device=self.device)
        self.assertEqual(output, expected, rtol=rtol, atol=atol)

    @skipIfNoExec("apply-cmvn-sliding")
    def test_sliding_window_cmn(self):
        """sliding_window_cmn should be numerically compatible with apply-cmvn-sliding"""
        kwargs = {
            "cmn_window": 600,
            "min_cmn_window": 100,
            "center": False,
            "norm_vars": False,
        }

        tensor = torch.randn(40, 10, dtype=self.dtype, device=self.device)
        result = F.sliding_window_cmn(tensor, **kwargs)
        command = ["apply-cmvn-sliding"] + convert_args(**kwargs) + ["ark:-", "ark:-"]
        kaldi_result = run_kaldi(command, "ark", tensor)
        self.assert_equal(result, expected=kaldi_result)
