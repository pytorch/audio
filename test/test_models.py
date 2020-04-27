import pytest

import torch
from torchaudio.models import Wav2Letter


class TestWav2Letter:
    @pytest.mark.parametrize('batch_size', [2])
    @pytest.mark.parametrize('num_features', [1])
    @pytest.mark.parametrize('num_classes', [40])
    @pytest.mark.parametrize('input_length', [320])
    def test_waveform(self, batch_size, num_features, num_classes, input_length):
        model = Wav2Letter()

        x = torch.rand(batch_size, num_features, input_length)
        out = model(x)

        assert out.size() == (batch_size, num_classes, 2)

    @pytest.mark.parametrize('batch_size', [2])
    @pytest.mark.parametrize('num_features', [13])
    @pytest.mark.parametrize('num_classes', [40])
    @pytest.mark.parametrize('input_length', [2])
    def test_mfcc(self, batch_size, num_features, num_classes, input_length):
        model = Wav2Letter(input_type="mfcc", num_features=13)

        x = torch.rand(batch_size, num_features, input_length)
        out = model(x)

        assert out.size() == (batch_size, num_classes, 2)
