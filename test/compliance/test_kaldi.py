import torchaudio.compliance.kaldi as kaldi
import torch
import unittest

class Test_Kaldi(unittest.TestCase):
    def _test_get_strided_helper(self, num_samples, window_size, window_shift, snip_edges):
        print(num_samples, window_size, window_shift, snip_edges)
        waveform = torch.arange(num_samples)
        res = kaldi._get_strided(waveform, window_size, window_shift, snip_edges)

    def test_get_strided(self):
        # generate any combination where 0 < window_size <= num_samples and
        # 0 < window_shift.

        for num_samples in range(1, 20):
            for window_size in range(1, num_samples + 1):
                for window_shift in range(1, 2*num_samples+1):
                    for snip_edges in range(0, 2):
                        self._test_get_strided_helper(num_samples, window_size, window_shift, snip_edges)



if __name__ == '__main__':
    unittest.main()
