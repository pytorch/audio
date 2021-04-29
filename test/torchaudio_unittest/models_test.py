import itertools
from collections import namedtuple

import torch
from parameterized import parameterized
from torchaudio.models import ConvTasNet, Wav2Letter, WaveRNN
from torchaudio.models.wavernn import MelResNet, UpsampleNetwork
from torchaudio_unittest import common_utils


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


_ConvTasNetParams = namedtuple(
    '_ConvTasNetParams',
    [
        'enc_num_feats',
        'enc_kernel_size',
        'msk_num_feats',
        'msk_num_hidden_feats',
        'msk_kernel_size',
        'msk_num_layers',
        'msk_num_stacks',
    ]
)


class TestConvTasNet(common_utils.TorchaudioTestCase):
    @parameterized.expand(list(itertools.product(
        [2, 3],
        [
            _ConvTasNetParams(128, 40, 128, 256, 3, 7, 2),
            _ConvTasNetParams(256, 40, 128, 256, 3, 7, 2),
            _ConvTasNetParams(512, 40, 128, 256, 3, 7, 2),
            _ConvTasNetParams(512, 40, 128, 256, 3, 7, 2),
            _ConvTasNetParams(512, 40, 128, 512, 3, 7, 2),
            _ConvTasNetParams(512, 40, 128, 512, 3, 7, 2),
            _ConvTasNetParams(512, 40, 256, 256, 3, 7, 2),
            _ConvTasNetParams(512, 40, 256, 512, 3, 7, 2),
            _ConvTasNetParams(512, 40, 256, 512, 3, 7, 2),
            _ConvTasNetParams(512, 40, 128, 512, 3, 6, 4),
            _ConvTasNetParams(512, 40, 128, 512, 3, 4, 6),
            _ConvTasNetParams(512, 40, 128, 512, 3, 8, 3),
            _ConvTasNetParams(512, 32, 128, 512, 3, 8, 3),
            _ConvTasNetParams(512, 16, 128, 512, 3, 8, 3),
        ],
    )))
    def test_paper_configuration(self, num_sources, model_params):
        """ConvTasNet model works on the valid configurations in the paper"""
        batch_size = 32
        num_frames = 8000

        model = ConvTasNet(
            num_sources=num_sources,
            enc_kernel_size=model_params.enc_kernel_size,
            enc_num_feats=model_params.enc_num_feats,
            msk_kernel_size=model_params.msk_kernel_size,
            msk_num_feats=model_params.msk_num_feats,
            msk_num_hidden_feats=model_params.msk_num_hidden_feats,
            msk_num_layers=model_params.msk_num_layers,
            msk_num_stacks=model_params.msk_num_stacks,
        )
        tensor = torch.rand(batch_size, 1, num_frames)
        output = model(tensor)

        assert output.shape == (batch_size, num_sources, num_frames)
