
import torch
from torchaudio.models.tacotron2 import _Tacotron2, _Encoder, _Decoder
from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    skipIfNoCuda,
    TempDirMixin,
)
from parameterized import parameterized


class TorchscriptConsistencyMixin(TempDirMixin):
    """Mixin to provide easy access assert torchscript consistency"""

    def _assert_torchscript_consistency(self, model, tensors):
        path = self.get_temp_path('func.zip')
        torch.jit.script(model).save(path)
        ts_func = torch.jit.load(path)

        torch.random.manual_seed(40)
        output = model(*tensors)

        torch.random.manual_seed(40)
        ts_output = ts_func(*tensors)

        self.assertEqual(ts_output, output)


class TestEncoder(TorchaudioTestCase, TorchscriptConsistencyMixin):

    def _get_model(self, encoder_n_convolutions=3, encoder_embedding_dim=512, encoder_kernel_size=5):
        return _Encoder(encoder_n_convolutions, encoder_embedding_dim, encoder_kernel_size)

    def _output_shape_test(self, model, n_batch, encoder_embedding_dim, n_seq, device, dtype):
        """Validate the output dimensions of a Encoder.
        """

        x = torch.rand(n_batch, encoder_embedding_dim, n_seq, device=device, dtype=dtype)
        input_lengths = torch.ones(n_batch, device=device, dtype=torch.int32) * n_seq
        out = model(x, input_lengths)

        assert out.size() == (n_batch, n_seq, encoder_embedding_dim)

    @parameterized.expand([(32, 64, 512, torch.float32, torch.device('cpu'))])
    def test_cpu_encoder_output(self, n_batch, n_seq, encoder_embedding_dim, dtype, device):
        model = self._get_model(encoder_embedding_dim=encoder_embedding_dim).to(device).eval()
        self._output_shape_test(model, n_batch, encoder_embedding_dim, n_seq, device, dtype)

    @parameterized.expand([(32, 64, 512, torch.float32, torch.device('cuda'))])
    @skipIfNoCuda
    def test_gpu_encoder_output(self, n_batch, n_seq, encoder_embedding_dim, dtype, device):
        model = self._get_model(encoder_embedding_dim=encoder_embedding_dim).to(device).eval()
        self._output_shape_test(model, n_batch, encoder_embedding_dim, n_seq, device, dtype)


class TestDecoder(TorchaudioTestCase, TorchscriptConsistencyMixin):

    def _get_model(self, n_mels=80, encoder_embedding_dim=512):
        model = _Decoder(n_mels=n_mels,
                         n_frames_per_step=1,
                         encoder_embedding_dim=encoder_embedding_dim,
                         attention_dim=128,
                         attention_rnn_dim=1024,
                         attention_location_n_filters=32,
                         attention_location_kernel_size=31,
                         decoder_rnn_dim=1024,
                         prenet_dim=256,
                         max_decoder_steps=2000,
                         gate_threshold=0.5,
                         p_attention_dropout=0.1,
                         p_decoder_dropout=0.1,
                         early_stopping=False)
        return model

    def _torchscript_consistency_test(self,
                                      model,
                                      n_mels,
                                      n_batch,
                                      encoder_embedding_dim,
                                      n_seq,
                                      device,
                                      dtype):
        """Validate the torchscript consistency of a Decoder.
        """

        n_time_steps = 200

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=dtype, device=device)
        decoder_inputs = torch.rand(n_batch, n_mels, n_time_steps, dtype=dtype, device=device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=device)

        self._assert_torchscript_consistency(model, (memory, decoder_inputs, memory_lengths))

    def _output_shape_test(self,
                           model,
                           n_mels,
                           n_batch,
                           encoder_embedding_dim,
                           n_seq,
                           device,
                           dtype):
        """Validate the output dimensions of a Decoder.
        """

        n_time_steps = 200

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=dtype, device=device)
        decoder_inputs = torch.rand(n_batch, n_mels, n_time_steps, dtype=dtype, device=device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=device)

        mel_outputs, gate_outputs, alignments = model(memory, decoder_inputs, memory_lengths)

        assert mel_outputs.size() == (n_batch, n_mels, n_time_steps)
        assert gate_outputs.size() == (n_batch, n_time_steps)
        assert alignments.size() == (n_batch, n_time_steps, n_seq)

    @parameterized.expand([(32, 80, 300, 512, torch.float32, torch.device('cpu'))])
    def test_cpu_torchscript_consistency(self, n_batch, n_mels, n_seq, encoder_embedding_dim, dtype, device):
        model = self._get_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim).to(device).eval()
        self._torchscript_consistency_test(model, n_mels, n_batch, encoder_embedding_dim, n_seq, device, dtype)

    @parameterized.expand([(32, 80, 300, 512, torch.float32, torch.device('cuda'))])
    @skipIfNoCuda
    def test_gpu_torchscript_consistency(self, n_batch, n_mels, n_seq, encoder_embedding_dim, dtype, device):
        model = self._get_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim).to(device).eval()
        self._torchscript_consistency_test(model, n_mels, n_batch, encoder_embedding_dim, n_seq, device, dtype)

    @parameterized.expand([(32, 80, 300, 512, torch.float32, torch.device('cpu'))])
    def test_cpu_decoder_output(self, n_batch, n_mels, n_seq, encoder_embedding_dim, dtype, device):
        model = self._get_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim).to(device).eval()
        self._output_shape_test(model, n_mels, n_batch, encoder_embedding_dim, n_seq, device, dtype)

    @parameterized.expand([(32, 80, 300, 512, torch.float32, torch.device('cuda'))])
    @skipIfNoCuda
    def test_gpu_decoder_output(self, n_batch, n_mels, n_seq, encoder_embedding_dim, dtype, device):
        model = self._get_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim).to(device).eval()
        self._output_shape_test(model, n_mels, n_batch, encoder_embedding_dim, n_seq, device, dtype)


class TestTacotron2(TorchaudioTestCase, TorchscriptConsistencyMixin):

    def _get_model(self, n_mels=80):
        model = _Tacotron2(
            mask_padding=False,
            n_mels=n_mels,
            n_symbols=148,
            symbols_embedding_dim=512,
            encoder_kernel_size=5,
            encoder_n_convolutions=3,
            encoder_embedding_dim=512,
            attention_rnn_dim=1024,
            attention_dim=128,
            attention_location_n_filters=32,
            attention_location_kernel_size=31,
            n_frames_per_step=1,
            decoder_rnn_dim=1024,
            prenet_dim=256,
            max_decoder_steps=2000,
            gate_threshold=0.5,
            p_attention_dropout=0.1,
            p_decoder_dropout=0.1,
            postnet_embedding_dim=512,
            postnet_kernel_size=5,
            postnet_n_convolutions=5,
            decoder_no_early_stopping=False)
        return model

    def _output_shape_test(self,
                           model,
                           n_mels,
                           n_batch,
                           device,
                           dtype):
        """Validate the output dimensions of a Tacotron2 model.
        """

        max_mel_specgram_length = 500
        max_text_length = 100

        text = torch.randint(0, 148, (n_batch, max_text_length), dtype=torch.int32, device=device)
        text_lengths = max_text_length * torch.ones([n_batch, ], dtype=torch.int32, device=device)
        mel_specgram = torch.rand(n_batch, n_mels, max_mel_specgram_length, dtype=dtype, device=device)
        mel_specgram_lengths = max_mel_specgram_length * torch.ones([n_batch, ], dtype=torch.int32, device=device)

        mel_out, mel_out_postnet, gate_outputs, alignments = model(
            text, text_lengths, mel_specgram, mel_specgram_lengths)

        assert mel_out.size() == (n_batch, n_mels, max_mel_specgram_length)
        assert mel_out_postnet.size() == (n_batch, n_mels, max_mel_specgram_length)
        assert gate_outputs.size() == (n_batch, max_mel_specgram_length)
        assert alignments.size() == (n_batch, max_mel_specgram_length, max_text_length)

    @parameterized.expand([(32, 80, torch.float32, torch.device('cpu'))])
    def test_cpu_decoder_output(self, n_batch, n_mels, dtype, device):
        model = self._get_model(n_mels=n_mels).to(device).eval()
        self._output_shape_test(model, n_mels, n_batch, device, dtype)

    @parameterized.expand([(32, 80, torch.float32, torch.device('cuda'))])
    @skipIfNoCuda
    def test_gpu_decoder_output(self, n_batch, n_mels, dtype, device):
        model = self._get_model(n_mels=n_mels).to(device).eval()
        self._output_shape_test(model, n_mels, n_batch, device, dtype)
