import torch
from torchaudio.models import Wav2Letter, _MelResNet, _UpsampleNetwork, _WaveRNN

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
        n_output_melresnet = 128
        n_res_block = 10
        n_hidden_resblock = 128
        kernel_size = 5

        model = _MelResNet(n_res_block, n_freq, n_hidden_resblock, n_output_melresnet, kernel_size)

        x = torch.rand(n_batch, n_freq, n_time)
        out = model(x)

        assert out.size() == (n_batch, n_output_melresnet, n_time - kernel_size + 1)


class TestUpsampleNetwork(common_utils.TorchaudioTestCase):

    def test_waveform(self):
        """Validate the output dimensions of a _UpsampleNetwork block.
        """

        upsample_scales = [5, 5, 8]
        n_batch = 2
        n_time = 200
        n_freq = 100
        n_output_melresnet = 256
        n_res_block = 10
        n_hidden_resblock = 128
        kernel_size = 5

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale

        model = _UpsampleNetwork(upsample_scales,
                                 n_res_block,
                                 n_freq,
                                 n_hidden_resblock,
                                 n_output_melresnet,
                                 kernel_size)

        x = torch.rand(n_batch, n_freq, n_time)
        out1, out2 = model(x)

        assert out1.size() == (n_batch, n_freq, total_scale * (n_time - kernel_size + 1))
        assert out2.size() == (n_batch, n_output_melresnet, total_scale * (n_time - kernel_size + 1))


class TestWaveRNN(common_utils.TorchaudioTestCase):

    def test_waveform(self):
        """Validate the output dimensions of a _WaveRNN model in waveform mode.
        """

        upsample_scales = [5, 5, 8]
        n_rnn = 512
        n_fc = 512
        n_classes = 512
        sample_rate = 24000
        hop_length = 200
        n_batch = 2
        n_time = 200
        n_freq = 100
        n_output_melresnet = 256
        n_res_block = 10
        n_hidden_resblock = 128
        kernel_size = 5
        loss = 'waveform'

        model = _WaveRNN(upsample_scales, n_classes, sample_rate, hop_length, n_res_block,
                         n_rnn, n_fc, kernel_size, n_freq, n_hidden_resblock, n_output_melresnet, loss)

        x = torch.rand(n_batch, 1, hop_length * (n_time - kernel_size + 1))
        mels = torch.rand(n_batch, 1, n_freq, n_time)
        out = model(x, mels)

        assert out.size() == (n_batch, 1, hop_length * (n_time - kernel_size + 1), n_classes)

    def test_mol(self):
        """Validate the output dimensions of a _WaveRNN model in mol mode.
        """

        upsample_scales = [5, 5, 8]
        n_rnn = 512
        n_fc = 512
        n_classes = 512
        sample_rate = 24000
        hop_length = 200
        n_batch = 2
        n_time = 200
        n_freq = 100
        n_output_melresnet = 256
        n_res_block = 10
        n_hidden_resblock = 128
        kernel_size = 5
        loss = 'mol'

        model = _WaveRNN(upsample_scales, n_classes, sample_rate, hop_length, n_res_block,
                         n_rnn, n_fc, kernel_size, n_freq, n_hidden_resblock, n_output_melresnet, loss)

        x = torch.rand(n_batch, 1, hop_length * (n_time - kernel_size + 1))
        mels = torch.rand(n_batch, 1, n_freq, n_time)
        out = model(x, mels)

        assert out.size() == (n_batch, 1, hop_length * (n_time - kernel_size + 1), 30)
