import os
import torch
import torchaudio.kaldi_io as kio
import unittest
import test.common_utils


class KaldiIOTest(unittest.TestCase):
    data1 = [[1, 2, 3], [11, 12, 13], [21, 22, 23]]
    data2 = [[31, 32, 33], [41, 42, 43], [51, 52, 53]]
    test_dirpath, test_dir = test.common_utils.create_temp_assets_dir()

    def _test_helper(self, file_name, expected_data, fn, expected_dtype):
        """ Takes a file_name to the input data and a function fn to extract the
        data. It compares the extracted data to the expected_data. The expected_dtype
        will be used to check that the extracted data is of the right type.
        """
        test_filepath = os.path.join(self.test_dirpath, "assets", file_name)
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


if __name__ == '__main__':
    unittest.main()
