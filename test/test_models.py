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

        batch_size = 2
        num_features = 200
        input_dims = 100
        output_dims = 128
        res_blocks = 10
        hidden_dims = 128
        pad = 2

        model = _MelResNet(res_blocks, input_dims, hidden_dims, output_dims, pad)

        x = torch.rand(batch_size, input_dims, num_features)
        out = model(x)

        assert out.size() == (batch_size, output_dims, num_features - pad * 2)
