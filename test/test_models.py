from __future__ import absolute_import, division, print_function, unicode_literals
import unittest

import torch
from torchaudio.models import wav2letter


class ModelTester(unittest.TestCase):
    def test_wav2letter(self):
        batch_size = 2
        n_features = 1
        input_length = 800

        model = wav2letter()
        x = torch.rand(batch_size, n_features, input_length)
        out = model(x)

        assert out.size() == (1, batch_size, 40)


if __name__ == '__main__':
    unittest.main()
