import torch
from torchaudio.models import Wav2Letter, MelResNet, UpsampleNetwork, WaveRNN

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
        """Validate the output dimensions of a MelResNet block.
        """

        n_batch = 2
        n_time = 200
        n_freq = 100
        n_output = 128
        n_res_block = 10
        n_hidden = 128
        kernel_size = 5

        model = MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)

        x = torch.rand(n_batch, n_freq, n_time)
        out = model(x)

        assert out.size() == (n_batch, n_output, n_time - kernel_size + 1)


class TestUpsampleNetwork(common_utils.TorchaudioTestCase):

    def test_waveform(self):
        """Validate the output dimensions of a UpsampleNetwork block.
        """

        upsample_scales = [5, 5, 8]
        n_batch = 2
        n_time = 200
        n_freq = 100
        n_output = 256
        n_res_block = 10
        n_hidden = 128
        kernel_size = 5

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale

        model = UpsampleNetwork(upsample_scales,
                                n_res_block,
                                n_freq,
                                n_hidden,
                                n_output,
                                kernel_size)

        x = torch.rand(n_batch, n_freq, n_time)
        out1, out2 = model(x)

        assert out1.size() == (n_batch, n_freq, total_scale * (n_time - kernel_size + 1))
        assert out2.size() == (n_batch, n_output, total_scale * (n_time - kernel_size + 1))


class TestWaveRNN(common_utils.TorchaudioTestCase):

    def test_waveform(self):
        """Validate the output dimensions of a WaveRNN model.
        """

        upsample_scales = [5, 5, 8]
        n_rnn = 512
        n_fc = 512
        n_classes = 512
        hop_length = 200
        n_batch = 2
        n_time = 200
        n_freq = 100
        n_output = 256
        n_res_block = 10
        n_hidden = 128
        kernel_size = 5

        model = WaveRNN(upsample_scales, n_classes, hop_length, n_res_block,
                        n_rnn, n_fc, kernel_size, n_freq, n_hidden, n_output)

        x = torch.rand(n_batch, 1, hop_length * (n_time - kernel_size + 1))
        mels = torch.rand(n_batch, 1, n_freq, n_time)
        out = model(x, mels)

        assert out.size() == (n_batch, 1, hop_length * (n_time - kernel_size + 1), n_classes)


class TestEncoder(common_utils.TorchaudioTestCase):
    def test_output(self):
        """Validate the output dimensions of a _Encoder block.
        """

        n_encoder_convolutions = 3
        n_encoder_embedding = 512
        n_encoder_kernel_size = 5
        n_batch = 32
        n_seq = 64

        model = _Encoder(n_encoder_convolutions, n_encoder_embedding, n_encoder_kernel_size)

        x = torch.rand(n_batch, n_encoder_embedding, n_seq)
        input_length = [n_seq for i in range(n_batch)]
        out = model(x, input_length)

        assert out.size() == (n_batch, n_seq, n_encoder_embedding)
    