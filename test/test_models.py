import unittest

import torch
from torchaudio.models import Wav2Letter


class ModelTester(unittest.TestCase):
    def test_wav2letter(self):
        batch_size = 2
        n_features = 1
        input_length = 320

        model = Wav2Letter()
        x = torch.rand(batch_size, n_features, input_length)
        out = model(x)

        assert out.size() == (2, batch_size, 40)


if __name__ == '__main__':
    unittest.main()
