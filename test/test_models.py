import torch
from torchaudio.models import Wav2Letter, MelResNet


class TestWav2Letter:

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


class TestMelResNet:
    @pytest.mark.parametrize('batch_size', [2])
    @pytest.mark.parametrize('num_features', [200])
    @pytest.mark.parametrize('input_dims', [100])
    @pytest.mark.parametrize('output_dims', [128])
    def test_waveform(self, batch_size, num_features, input_dims, output_dims):
        model = MelResNet()

        x = torch.rand(batch_size, input_dims, num_features)
        out = model(x)

        assert out.size() == (batch_size, output_dims, num_features - 4)
