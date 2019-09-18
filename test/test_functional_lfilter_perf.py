from __future__ import absolute_import, division, print_function, unicode_literals
import math
import os
import torch
import torchaudio
import unittest
import common_utils

from torchaudio.functional import lfilter
from _torch_filtering import _lfilter_tensor_matrix, _lfilter_element_wise


class TestFunctionalLFilterPerformance(unittest.TestCase):
    test_dirpath, test_dir = common_utils.create_temp_assets_dir()

    @staticmethod
    def run_test(n_channels, n_frames, n_order_filter, assertClose=True):
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
        elif n_order_filter == 18:
            # >>> import scipy.signal
            # >>> wp = 0.48, ws = 0.5, gpass = 0.2, gstop = 120
            # >>> b, a = scipy.signal.iirdesign(wp, ws, gpass, gstop)
            b_coeffs = [
                0.0006050813536446144,
                0.002920916369302935,
                0.010247568347759453,
                0.02591236698507957,
                0.05390501051935878,
                0.09344581172781004,
                0.13951533321139883,
                0.1808658576803922,
                0.2056643061895918,
                0.2056643061895911,
                0.1808658576803912,
                0.13951533321139847,
                0.09344581172781012,
                0.053905010519358885,
                0.02591236698507962,
                0.010247568347759466,
                0.0029209163693029367,
                0.0006050813536446148,
            ]
            a_coeffs = [
                1.0,
                -4.3964136877356745,
                14.650181359641305,
                -34.45816395187684,
                67.18247518997862,
                -108.01956225077998,
                149.4332056661277,
                -178.07791467502364,
                185.28267044557634,
                -168.13382659655514,
                133.22364764531704,
                -91.59439958870928,
                54.15835239046956,
                -27.090521914173934,
                11.163677645454127,
                -3.627296054625132,
                0.8471764313073272,
                -0.11712354962357388,
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
        print(
            "lfilter perf - Data Size: [%d x %d], Filter Order: %d"
            % (waveform.size(0), waveform.size(1), a_coeffs.size(0))
        )
        print("-" * 80)
        print("Python Matrix Runtime [current]: %10.6f s" % run_time_1)
        print("CPP Element Wise Runtime       : %10.6f s" % run_time_2)
        print("CPP Matrix Runtime             : %10.6f s" % run_time_3)
        print("-" * 80)
        print("Ratio Python / CPP ElementWise : %10.2f x" % (run_time_1 / run_time_2))

        if assertClose:
            # maxDeviation = torch.kthvalue(torch.abs(output_waveform_3- output_waveform_2), output_waveform_1.size(1))
            assert torch.allclose(output_waveform_1, output_waveform_2, atol=3e-4)
            assert torch.allclose(output_waveform_2, output_waveform_3, atol=3e-4)
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

        # For higher order filters, due to floating point precision
        #  matrix method and element method can get different results depending on order of operations
        # Also, for longer signals and higher filters, easier to create unstable filter
        # https://dsp.stackexchange.com/questions/54386/relation-between-order-and-stability-in-iir-filter
        self.run_test(2, 8000, 18, False)
        self.run_test(2, 80000, 18, False)
        self.run_test(2, 800000, 18, False)


if __name__ == "__main__":
    unittest.main()
