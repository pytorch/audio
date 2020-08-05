import torch
from torchaudio.models import Wav2Letter, MelResNet, UpsampleNetwork, WaveRNN, _Encoder, _Decoder, _Tacotron2

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


class TestDecoder(common_utils.TorchaudioTestCase):
    def test_output(self):
        """Validate the output dimensions of a _Decoder block.
        """

        n_mel_channels = 80
        n_frames_per_step = 1
        n_encoder_embedding = 512
        n_attention = 128
        attention_location_n_filters = 32
        attention_location_kernel_size = 31
        n_attention_rnn = 1024
        n_decoder_rnn = 1024
        n_prenet = 256
        max_decoder_steps = 2000
        gate_threshold = 0.5
        p_attention_dropout = 0.1
        p_decoder_dropout = 0.1
        early_stopping = False
        n_batch = 32
        T_out = 200
        n_seq = 300

        model = _Decoder(n_mel_channels,
                         n_frames_per_step,
                         n_encoder_embedding,
                         n_attention,
                         attention_location_n_filters,
                         attention_location_kernel_size,
                         n_attention_rnn,
                         n_decoder_rnn,
                         n_prenet,
                         max_decoder_steps,
                         gate_threshold,
                         p_attention_dropout,
                         p_decoder_dropout,
                         early_stopping)

        memory = torch.rand(n_batch, n_seq, n_encoder_embedding)
        decoder_inputs = torch.rand(n_batch, n_mel_channels, T_out)
        memory_lengths = torch.ones(n_batch,)

        mel_outputs, gate_outputs, alignments = model(memory, decoder_inputs, memory_lengths)

        assert mel_outputs.size() == (n_batch, n_mel_channels, T_out)
        assert gate_outputs.size() == (n_batch, T_out)
        assert alignments.size() == (n_batch, T_out, n_seq)


class TestTacotron2(common_utils.TorchaudioTestCase):
    def test_output(self):
        """Validate the output dimensions of a _Tacotron2 block.
        """

        mask_padding = False
        n_mel_channels = 80
        n_symbols = 100
        n_symbols_embedding = 512
        encoder_kernel_size = 5
        encoder_n_convolutions = 3
        n_encoder_embedding = 512
        n_attention = 128
        attention_location_n_filters = 32
        attention_location_kernel_size = 31
        n_frames_per_step = 1
        n_attention_rnn = 1024
        n_decoder_rnn = 1024
        n_prenet = 256
        max_decoder_steps = 2000
        gate_threshold = 0.5
        p_attention_dropout = 0.1
        p_decoder_dropout = 0.1
        n_postnet_embedding = 512
        postnet_kernel_size = 5
        postnet_n_convolutions = 5
        decoder_no_early_stopping = True

        n_batch = 32
        max_target_len = 500
        n_seq = 100
        max_len = 10

        model = _Tacotron2(mask_padding,
                           n_mel_channels,
                           n_symbols,
                           n_symbols_embedding,
                           encoder_kernel_size,
                           encoder_n_convolutions,
                           n_encoder_embedding,
                           n_attention_rnn,
                           n_attention,
                           attention_location_n_filters,
                           attention_location_kernel_size,
                           n_frames_per_step,
                           n_decoder_rnn,
                           n_prenet,
                           max_decoder_steps,
                           gate_threshold,
                           p_attention_dropout,
                           p_decoder_dropout,
                           n_postnet_embedding,
                           postnet_kernel_size,
                           postnet_n_convolutions,
                           decoder_no_early_stopping)

        text_padded = torch.randint(0, 30, (n_batch, n_seq))
        input_lengths = max_len * torch.ones([n_batch, ], dtype=torch.int32)
        output_lengths = 20 * torch.ones([n_batch, ], dtype=torch.int32)
        mel_padded = torch.rand(n_batch, n_mel_channels, max_target_len)

        inputs = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
        mel_out, mel_out_postnet, gate_outputs, alignments = model(inputs)

        assert mel_out.size() == (n_batch, n_mel_channels, max_target_len)
        assert mel_out_postnet.size() == (n_batch, n_mel_channels, max_target_len)
        assert gate_outputs.size() == (n_batch, max_target_len)
        assert alignments.size() == (n_batch, max_target_len, max_len)
