import unittest

import torch
import torchaudio.kaldi_io as kio

from torchaudio_unittest import common_utils


class Test_KaldiIO(common_utils.TorchaudioTestCase):
    data1 = [[1, 2, 3], [11, 12, 13], [21, 22, 23]]
    data2 = [[31, 32, 33], [41, 42, 43], [51, 52, 53]]

    def _test_helper(self, file_name, expected_data, fn, expected_dtype):
        """ Takes a file_name to the input data and a function fn to extract the
        data. It compares the extracted data to the expected_data. The expected_dtype
        will be used to check that the extracted data is of the right type.
        """
        test_filepath = common_utils.get_asset_path(file_name)
        expected_output = {'key' + str(idx + 1): torch.tensor(val, dtype=expected_dtype)
                           for idx, val in enumerate(expected_data)}

        for key, vec in fn(test_filepath):
            self.assertTrue(key in expected_output)
            self.assertTrue(isinstance(vec, torch.Tensor))
            self.assertEqual(vec.dtype, expected_dtype)
            self.assertTrue(torch.all(torch.eq(vec, expected_output[key])))

    def test_read_vec_int_ark(self):
        self._test_helper("vec_int.ark", self.data1, kio.read_vec_int_ark, torch.int32)

    def test_read_vec_flt_ark(self):
        self._test_helper("vec_flt.ark", self.data1, kio.read_vec_flt_ark, torch.float32)

    def test_read_mat_ark(self):
        self._test_helper("mat.ark", [self.data1, self.data2], kio.read_mat_ark, torch.float32)
