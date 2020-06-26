import torch
from torchaudio.models import Wav2Letter, _MelResNet

from . import common_utils


class TestWav2Letter(common_utils.TorchaudioTestCase):

    def test_waveform(self):
        batch_size = 2
        num_features = 1
        num_classes = 40
        input_length = 320

        model = Wav2Letter(num_classes=num_classes, num_features=num_features)

        x = torch.rand(batch_size, num_features, input_length)
        out = model(x)

        assert out.size() == (batch_size, num_classes, 2)

    def test_mfcc(self):
        batch_size = 2
        num_features = 13
        num_classes = 40
        input_length = 2

        model = Wav2Letter(num_classes=num_classes, input_type="mfcc", num_features=num_features)

        x = torch.rand(batch_size, num_features, input_length)
        out = model(x)

        assert out.size() == (batch_size, num_classes, 2)


class TestMelResNet(common_utils.TorchaudioTestCase):

    def test_waveform(self):
        """Validate the output dimensions of a _MelResNet block.
        """

        n_batch = 2
        n_time = 200
        n_freq = 100
        n_output = 256
        n_res_block = 10
        n_hidden = 128
        kernel_size = 5

        model = _MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)

        x = torch.rand(n_batch, n_freq, n_time)
        out = model(x)

        assert out.size() == (n_batch, n_output, n_time - kernel_size + 1)
