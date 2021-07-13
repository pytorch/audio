import torch
from torchaudio.prototype.tacotron2 import Tacotron2, _Encoder, _Decoder
from torchaudio_unittest.common_utils import (
    TestBaseMixin,
    TempDirMixin,
)


class TorchscriptConsistencyMixin(TempDirMixin):
    r"""Mixin to provide easy access assert torchscript consistency"""

    def _assert_torchscript_consistency(self, model, tensors):
        path = self.get_temp_path("func.zip")
        torch.jit.script(model).save(path)
        ts_func = torch.jit.load(path)

        torch.random.manual_seed(40)
        output = model(*tensors)

        torch.random.manual_seed(40)
        ts_output = ts_func(*tensors)

        self.assertEqual(ts_output, output)


class Tacotron2EncoderTests(TestBaseMixin, TorchscriptConsistencyMixin):
    def test_tacotron2_torchscript_consistency(self):
        r"""Validate the torchscript consistency of a Encoder."""
        n_batch, n_seq, encoder_embedding_dim = 16, 64, 512
        model = (
            _Encoder(
                encoder_embedding_dim=encoder_embedding_dim,
                encoder_n_convolution=3,
                encoder_kernel_size=5,
            )
            .to(self.device)
            .eval()
        )

        x = torch.rand(
            n_batch, encoder_embedding_dim, n_seq, device=self.device, dtype=self.dtype
        )
        input_lengths = (
            torch.ones(n_batch, device=self.device, dtype=torch.int32) * n_seq
        )

        self._assert_torchscript_consistency(model, (x, input_lengths))

    def test_encoder_output_shape(self):
        r"""Feed tensors with specific shape to Tacotron2 Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        n_batch, n_seq, encoder_embedding_dim = 16, 64, 512
        model = (
            _Encoder(
                encoder_embedding_dim=encoder_embedding_dim,
                encoder_n_convolution=3,
                encoder_kernel_size=5,
            )
            .to(self.device)
            .eval()
        )

        x = torch.rand(
            n_batch, encoder_embedding_dim, n_seq, device=self.device, dtype=self.dtype
        )
        input_lengths = (
            torch.ones(n_batch, device=self.device, dtype=torch.int32) * n_seq
        )
        out = model(x, input_lengths)

        assert out.size() == (n_batch, n_seq, encoder_embedding_dim)


def _get_decoder_model(n_mel=80, encoder_embedding_dim=512):
    model = _Decoder(
        n_mel=n_mel,
        n_frames_per_step=1,
        encoder_embedding_dim=encoder_embedding_dim,
        decoder_rnn_dim=1024,
        decoder_max_step=2000,
        decoder_dropout=0.1,
        decoder_early_stopping=False,
        attention_rnn_dim=1024,
        attention_hidden_dim=128,
        attention_location_n_filter=32,
        attention_location_kernel_size=31,
        attention_dropout=0.1,
        prenet_dim=256,
        gate_threshold=0.5,
    )
    return model


class Tacotron2DecoderTests(TestBaseMixin, TorchscriptConsistencyMixin):
    def test_decoder_torchscript_consistency(self):
        r"""Validate the torchscript consistency of a Decoder."""
        n_batch = 16
        n_mel = 80
        n_seq = 200
        encoder_embedding_dim = 256
        n_time_steps = 150

        model = (
            _get_decoder_model(n_mel=n_mel, encoder_embedding_dim=encoder_embedding_dim)
            .to(self.device)
            .eval()
        )

        memory = torch.rand(
            n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device
        )
        decoder_inputs = torch.rand(
            n_batch, n_mel, n_time_steps, dtype=self.dtype, device=self.device
        )
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        self._assert_torchscript_consistency(
            model, (memory, decoder_inputs, memory_lengths)
        )

    def test_decoder_output_shape(self):
        r"""Feed tensors with specific shape to Tacotron2 Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        n_batch = 16
        n_mel = 80
        n_seq = 200
        encoder_embedding_dim = 256
        n_time_steps = 150

        model = (
            _get_decoder_model(n_mel=n_mel, encoder_embedding_dim=encoder_embedding_dim)
            .to(self.device)
            .eval()
        )

        memory = torch.rand(
            n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device
        )
        decoder_inputs = torch.rand(
            n_batch, n_mel, n_time_steps, dtype=self.dtype, device=self.device
        )
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        mel_outputs, gate_outputs, alignments = model(
            memory, decoder_inputs, memory_lengths
        )

        assert mel_outputs.size() == (n_batch, n_mel, n_time_steps)
        assert gate_outputs.size() == (n_batch, n_time_steps)
        assert alignments.size() == (n_batch, n_time_steps, n_seq)


def _get_tacotron2_model(n_mel):
    return Tacotron2(
        mask_padding=False,
        n_mel=n_mel,
        n_symbol=148,
        n_frames_per_step=1,
        symbol_embedding_dim=512,
        encoder_embedding_dim=512,
        encoder_n_convolution=3,
        encoder_kernel_size=5,
        decoder_rnn_dim=1024,
        decoder_max_step=2000,
        decoder_dropout=0.1,
        decoder_early_stopping=True,
        attention_rnn_dim=1024,
        attention_hidden_dim=128,
        attention_location_n_filter=32,
        attention_location_kernel_size=31,
        attention_dropout=0.1,
        prenet_dim=256,
        postnet_n_convolution=5,
        postnet_kernel_size=5,
        postnet_embedding_dim=512,
        gate_threshold=0.5,
    )


class Tacotron2Tests(TestBaseMixin, TorchscriptConsistencyMixin):
    def _get_inputs(
        self, n_mel, n_batch: int, max_mel_specgram_length: int, max_text_length: int
    ):
        text = torch.randint(
            0, 148, (n_batch, max_text_length), dtype=torch.int32, device=self.device
        )
        text_lengths = max_text_length * torch.ones(
            (n_batch,), dtype=torch.int32, device=self.device
        )
        mel_specgram = torch.rand(
            n_batch,
            n_mel,
            max_mel_specgram_length,
            dtype=self.dtype,
            device=self.device,
        )
        mel_specgram_lengths = max_mel_specgram_length * torch.ones(
            (n_batch,), dtype=torch.int32, device=self.device
        )
        return text, text_lengths, mel_specgram, mel_specgram_lengths

    def test_tacotron2_torchscript_consistency(self):
        r"""Validate the torchscript consistency of a Tacotron2."""
        n_batch = 16
        n_mel = 80
        max_mel_specgram_length = 300
        max_text_length = 100

        model = _get_tacotron2_model(n_mel).to(self.device).eval()
        inputs = self._get_inputs(
            n_mel, n_batch, max_mel_specgram_length, max_text_length
        )

        self._assert_torchscript_consistency(model, inputs)

    def test_tacotron2_output_shape(self):
        r"""Feed tensors with specific shape to Tacotron2 and validate
        that it outputs with a tensor with expected shape.
        """
        n_batch = 16
        n_mel = 80
        max_mel_specgram_length = 300
        max_text_length = 100

        model = _get_tacotron2_model(n_mel).to(self.device).eval()
        inputs = self._get_inputs(
            n_mel, n_batch, max_mel_specgram_length, max_text_length
        )
        mel_out, mel_out_postnet, gate_outputs, alignments = model(*inputs)

        assert mel_out.size() == (n_batch, n_mel, max_mel_specgram_length)
        assert mel_out_postnet.size() == (n_batch, n_mel, max_mel_specgram_length)
        assert gate_outputs.size() == (n_batch, max_mel_specgram_length)
        assert alignments.size() == (n_batch, max_mel_specgram_length, max_text_length)

    def test_tacotron2_backward(self):
        r"""Make sure calling the backward function on Tacotron2's outputs does
        not error out. Following: https://github.com/pytorch/vision/blob/23b8760374a5aaed53c6e5fc83a7e83dbe3b85df/test/test_models.py#L255
        """
        n_batch = 16
        n_mel = 80
        max_mel_specgram_length = 300
        max_text_length = 100

        model = _get_tacotron2_model(n_mel).to(self.device)
        inputs = self._get_inputs(
            n_mel, n_batch, max_mel_specgram_length, max_text_length
        )
        mel_out, mel_out_postnet, gate_outputs, _ = model(*inputs)

        mel_out.sum().backward(retain_graph=True)
        mel_out_postnet.sum().backward(retain_graph=True)
        gate_outputs.sum().backward()
