import torch
from torchaudio.prototype.tacotron2 import Tacotron2, _Encoder, _Decoder
from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    TempDirMixin,
)


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


class Tacotron2EncoderTests(TestBaseMixin, TorchscriptConsistencyMixin):

    def test_encoder_output_shape(self):
        """Feed tensors with specific shape to Tacotron2 Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        n_batch, n_seq, encoder_embedding_dim = 32, 64, 512
        model = _Encoder(
            encoder_n_convolutions=3,
            encoder_embedding_dim=encoder_embedding_dim,
            encoder_kernel_size=5
        ).to(self.device).eval()

        x = torch.rand(n_batch, encoder_embedding_dim, n_seq, device=self.device, dtype=self.dtype)
        input_lengths = torch.ones(n_batch, device=self.device, dtype=torch.int32) * n_seq
        out = model(x, input_lengths)

        assert out.size() == (n_batch, n_seq, encoder_embedding_dim)


class Tacotron2DecoderTests(TestBaseMixin, TorchscriptConsistencyMixin):

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

    def test_torchscript_consistency(self):
        """Validate the torchscript consistency of a Decoder.
        """
        n_batch, n_mels, n_seq, encoder_embedding_dim, n_time_steps = 32, 80, 300, 512, 200
        model = self._get_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim).to(self.device).eval()

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device)
        decoder_inputs = torch.rand(n_batch, n_mels, n_time_steps, dtype=self.dtype, device=self.device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        self._assert_torchscript_consistency(model, (memory, decoder_inputs, memory_lengths))

    def test_decoder_output_shape(self):
        """Feed tensors with specific shape to Tacotron2 Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        n_batch, n_mels, n_seq, encoder_embedding_dim, n_time_steps = 32, 80, 300, 512, 200
        model = self._get_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim).to(self.device).eval()

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device)
        decoder_inputs = torch.rand(n_batch, n_mels, n_time_steps, dtype=self.dtype, device=self.device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        mel_outputs, gate_outputs, alignments = model(memory, decoder_inputs, memory_lengths)

        assert mel_outputs.size() == (n_batch, n_mels, n_time_steps)
        assert gate_outputs.size() == (n_batch, n_time_steps)
        assert alignments.size() == (n_batch, n_time_steps, n_seq)


class Tacotron2Tests(TestBaseMixin, TorchscriptConsistencyMixin):

    def test_tacotron2_output_shape(self):
        """Feed tensors with specific shape to Tacotron2 and validate
        that it outputs with a tensor with expected shape.
        """

        n_batch, n_mels = 32, 80
        model = Tacotron2(
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
            decoder_no_early_stopping=False).to(self.device).eval()

        max_mel_specgram_length = 500
        max_text_length = 100

        text = torch.randint(0, 148, (n_batch, max_text_length), dtype=torch.int32, device=self.device)
        text_lengths = max_text_length * torch.ones([n_batch, ], dtype=torch.int32, device=self.device)
        mel_specgram = torch.rand(n_batch, n_mels, max_mel_specgram_length, dtype=self.dtype, device=self.device)
        mel_specgram_lengths = max_mel_specgram_length * torch.ones([n_batch, ], dtype=torch.int32, device=self.device)

        mel_out, mel_out_postnet, gate_outputs, alignments = model(
            text, text_lengths, mel_specgram, mel_specgram_lengths)

        assert mel_out.size() == (n_batch, n_mels, max_mel_specgram_length)
        assert mel_out_postnet.size() == (n_batch, n_mels, max_mel_specgram_length)
        assert gate_outputs.size() == (n_batch, max_mel_specgram_length)
        assert alignments.size() == (n_batch, max_mel_specgram_length, max_text_length)
