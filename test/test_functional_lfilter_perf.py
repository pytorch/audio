from __future__ import absolute_import, division, print_function, unicode_literals
import math
import os
import torch
import torchaudio
import unittest
import common_utils

from torchaudio.functional import lfilter
from _torch_filtering import (
    _lfilter_tensor_matrix,
    _lfilter_tensor,
    _lfilter_element_wise,
)


class TestFunctionalLFilterPerformance(unittest.TestCase):
    test_dirpath, test_dir = common_utils.create_temp_assets_dir()

    @staticmethod
    def run_test(n_channels, n_frames, n_order_filter):
        waveform = torch.rand(n_channels, n_frames, device="cpu")
        b_coeffs = torch.rand(n_order_filter, dtype=torch.float32, device="cpu")
        a_coeffs = torch.rand(n_order_filter, dtype=torch.float32, device="cpu")

        if n_order_filter == 8:
            # Eighth Order Filter
            #  >>> import scipy.signal
            #  >>>  wp = 0.3
            #  >>>  ws = 0.5
            #  >>>  gpass = 1
            #  >>>  gstop = 100
            #  >>>  b, a = scipy.signal.iirdesign(wp, ws, gpass, gstop)
            b_coeffs = [
                0.0006544487997063485,
                0.001669274889397942,
                0.003218714446315984,
                0.004222562499298002,
                0.004222562499298002,
                0.0032187144463159834,
                0.0016692748893979413,
                0.0006544487997063485,
            ]
            a_coeffs = [
                1.0,
                -4.67403506662255,
                10.516336803850786,
                -14.399207825856776,
                12.844181702707655,
                -7.43604712843608,
                2.5888616732696077,
                -0.4205601576432048,
            ]
        elif n_order_filter == 5:
            # Fifth Order Filter
            # >>> import scipy.signal
            # >>> wp = 0.3, ws = 0.5, gpass = 1, gstop = 40
            # >>> b, a = scipy.signal.iirdesign(wp, ws, gpass, gstop)
            b_coeffs = [
                0.0353100066384039,
                0.023370652985988206,
                0.0560524973457262,
                0.023370652985988193,
                0.03531000663840389,
            ]
            a_coeffs = [
                1.0,
                -2.321010052951366,
                2.677193357612127,
                -1.5774235418173692,
                0.4158137396065854,
            ]

        # Cast into Tensors
        a_coeffs = torch.tensor(a_coeffs, device="cpu", dtype=torch.float32)
        b_coeffs = torch.tensor(b_coeffs, device="cpu", dtype=torch.float32)

        def time_and_output(func):
            import time

            st = time.time()
            output = func()
            run_time = time.time() - st
            return (output, run_time)

        (output_waveform_1, run_time_1) = time_and_output(
            lambda: lfilter(waveform, a_coeffs, b_coeffs)
        )
        (output_waveform_2, run_time_2) = time_and_output(
            lambda: _lfilter_element_wise(waveform, a_coeffs, b_coeffs)
        )
        (output_waveform_3, run_time_3) = time_and_output(
            lambda: _lfilter_tensor_matrix(waveform, a_coeffs, b_coeffs)
        )

        print("-" * 80)
        print("Evaluating Runtime between lfilter implementations")
        print("-" * 80)
        print(
            "Data Size: [%d x %d], Filter Order: %d"
            % (waveform.size(0), waveform.size(1), a_coeffs.size(0))
        )
        print("-" * 80)
        print("Python Matrix Runtime [current]: %10.6f s" % run_time_1)
        print("CPP Element Wise Runtime       : %10.6f s" % run_time_2)
        print("CPP Matrix Runtime             : %10.6f s" % run_time_3)

        # maxDeviation = torch.kthvalue(torch.abs(output_waveform_1- output_waveform_2), output_waveform_1.size(1))

        assert torch.allclose(output_waveform_1, output_waveform_2, atol=3e-4)
        assert torch.allclose(output_waveform_1, output_waveform_3, atol=3e-4)
        print("-" * 80)
        print("✓ - all outputs are identical")
        print("-" * 80)

    def test_lfilter_cmp(self):
        """
        Runs comparison on CPU
        """

        torch.random.manual_seed(423)
        self.run_test(2, 8000, 5)
        self.run_test(2, 80000, 5)
        self.run_test(2, 800000, 5)
        self.run_test(2, 8000, 8)
        self.run_test(2, 80000, 8)
        self.run_test(2, 800000, 8)


if __name__ == "__main__":
    unittest.main()
