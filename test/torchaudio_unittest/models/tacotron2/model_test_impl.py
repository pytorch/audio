from typing import Tuple

import torch
from parameterized import parameterized
from torch import Tensor
from torchaudio.models import Tacotron2
from torchaudio.models.tacotron2 import _Decoder, _Encoder
from torchaudio_unittest.common_utils import skipIfPy310, TestBaseMixin, torch_script


class Tacotron2InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text: Tensor, text_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.model.infer(text, text_lengths)


class Tacotron2DecoderInferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, memory: Tensor, memory_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model.infer(memory, memory_lengths)


class TorchscriptConsistencyMixin(TestBaseMixin):
    r"""Mixin to provide easy access assert torchscript consistency"""

    def _assert_torchscript_consistency(self, model, tensors):
        ts_func = torch_script(model)

        torch.random.manual_seed(40)
        output = model(*tensors)

        torch.random.manual_seed(40)
        ts_output = ts_func(*tensors)

        self.assertEqual(ts_output, output)


class Tacotron2EncoderTests(TorchscriptConsistencyMixin):
    @skipIfPy310
    def test_tacotron2_torchscript_consistency(self):
        r"""Validate the torchscript consistency of a Encoder."""
        n_batch, n_seq, encoder_embedding_dim = 16, 64, 512
        model = (
            _Encoder(encoder_embedding_dim=encoder_embedding_dim, encoder_n_convolution=3, encoder_kernel_size=5)
            .to(self.device)
            .eval()
        )

        x = torch.rand(n_batch, encoder_embedding_dim, n_seq, device=self.device, dtype=self.dtype)
        input_lengths = torch.ones(n_batch, device=self.device, dtype=torch.int32) * n_seq

        self._assert_torchscript_consistency(model, (x, input_lengths))

    def test_encoder_output_shape(self):
        r"""Feed tensors with specific shape to Tacotron2 Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        n_batch, n_seq, encoder_embedding_dim = 16, 64, 512
        model = (
            _Encoder(encoder_embedding_dim=encoder_embedding_dim, encoder_n_convolution=3, encoder_kernel_size=5)
            .to(self.device)
            .eval()
        )

        x = torch.rand(n_batch, encoder_embedding_dim, n_seq, device=self.device, dtype=self.dtype)
        input_lengths = torch.ones(n_batch, device=self.device, dtype=torch.int32) * n_seq
        out = model(x, input_lengths)

        assert out.size() == (n_batch, n_seq, encoder_embedding_dim)


def _get_decoder_model(n_mels=80, encoder_embedding_dim=512, decoder_max_step=2000, gate_threshold=0.5):
    model = _Decoder(
        n_mels=n_mels,
        n_frames_per_step=1,
        encoder_embedding_dim=encoder_embedding_dim,
        decoder_rnn_dim=1024,
        decoder_max_step=decoder_max_step,
        decoder_dropout=0.1,
        decoder_early_stopping=True,
        attention_rnn_dim=1024,
        attention_hidden_dim=128,
        attention_location_n_filter=32,
        attention_location_kernel_size=31,
        attention_dropout=0.1,
        prenet_dim=256,
        gate_threshold=gate_threshold,
    )
    return model


class Tacotron2DecoderTests(TorchscriptConsistencyMixin):
    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    def test_decoder_torchscript_consistency(self, n_batch):
        r"""Validate the torchscript consistency of a Decoder."""
        n_mels = 80
        n_seq = 200
        encoder_embedding_dim = 256
        n_time_steps = 150

        model = _get_decoder_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim)
        model = model.to(self.device).eval()

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device)
        decoder_inputs = torch.rand(n_batch, n_mels, n_time_steps, dtype=self.dtype, device=self.device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        self._assert_torchscript_consistency(model, (memory, decoder_inputs, memory_lengths))

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    def test_decoder_output_shape(self, n_batch):
        r"""Feed tensors with specific shape to Tacotron2 Decoder and validate
        that it outputs with a tensor with expected shape.
        """
        n_mels = 80
        n_seq = 200
        encoder_embedding_dim = 256
        n_time_steps = 150

        model = _get_decoder_model(n_mels=n_mels, encoder_embedding_dim=encoder_embedding_dim)
        model = model.to(self.device).eval()

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device)
        decoder_inputs = torch.rand(n_batch, n_mels, n_time_steps, dtype=self.dtype, device=self.device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        mel_specgram, gate_outputs, alignments = model(memory, decoder_inputs, memory_lengths)

        assert mel_specgram.size() == (n_batch, n_mels, n_time_steps)
        assert gate_outputs.size() == (n_batch, n_time_steps)
        assert alignments.size() == (n_batch, n_time_steps, n_seq)

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    def test_decoder_inference_torchscript_consistency(self, n_batch):
        r"""Validate the torchscript consistency of a Decoder."""
        n_mels = 80
        n_seq = 200
        encoder_embedding_dim = 256
        decoder_max_step = 300  # make inference more efficient
        gate_threshold = 0.505  # make inference more efficient

        model = _get_decoder_model(
            n_mels=n_mels,
            encoder_embedding_dim=encoder_embedding_dim,
            decoder_max_step=decoder_max_step,
            gate_threshold=gate_threshold,
        )
        model = model.to(self.device).eval()

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        model_wrapper = Tacotron2DecoderInferenceWrapper(model)

        self._assert_torchscript_consistency(model_wrapper, (memory, memory_lengths))

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    def test_decoder_inference_output_shape(self, n_batch):
        r"""Validate the torchscript consistency of a Decoder."""
        n_mels = 80
        n_seq = 200
        encoder_embedding_dim = 256
        decoder_max_step = 300  # make inference more efficient
        gate_threshold = 0.505  # if set to 0.5, the model will only run one step

        model = _get_decoder_model(
            n_mels=n_mels,
            encoder_embedding_dim=encoder_embedding_dim,
            decoder_max_step=decoder_max_step,
            gate_threshold=gate_threshold,
        )
        model = model.to(self.device).eval()

        memory = torch.rand(n_batch, n_seq, encoder_embedding_dim, dtype=self.dtype, device=self.device)
        memory_lengths = torch.ones(n_batch, dtype=torch.int32, device=self.device)

        mel_specgram, mel_specgram_lengths, gate_outputs, alignments = model.infer(memory, memory_lengths)

        assert len(mel_specgram.size()) == 3
        assert mel_specgram.size()[:-1] == (
            n_batch,
            n_mels,
        )
        assert mel_specgram.size()[2] == mel_specgram_lengths.max().item()
        assert len(mel_specgram_lengths.size()) == 1
        assert mel_specgram_lengths.size()[0] == n_batch
        assert mel_specgram_lengths.max().item() <= model.decoder_max_step
        assert len(gate_outputs.size()) == 2
        assert gate_outputs.size()[0] == n_batch
        assert gate_outputs.size()[1] == mel_specgram_lengths.max().item()
        assert len(alignments.size()) == 2
        assert alignments.size()[0] == n_seq
        assert alignments.size()[1] == mel_specgram_lengths.max().item() * n_batch


def _get_tacotron2_model(n_mels, decoder_max_step=2000, gate_threshold=0.5):
    return Tacotron2(
        mask_padding=False,
        n_mels=n_mels,
        n_symbol=148,
        n_frames_per_step=1,
        symbol_embedding_dim=512,
        encoder_embedding_dim=512,
        encoder_n_convolution=3,
        encoder_kernel_size=5,
        decoder_rnn_dim=1024,
        decoder_max_step=decoder_max_step,
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
        gate_threshold=gate_threshold,
    )


class Tacotron2Tests(TorchscriptConsistencyMixin):
    def _get_inputs(self, n_mels: int, n_batch: int, max_mel_specgram_length: int, max_text_length: int):
        text = torch.randint(0, 148, (n_batch, max_text_length), dtype=torch.int32, device=self.device)
        text_lengths = max_text_length * torch.ones((n_batch,), dtype=torch.int32, device=self.device)
        mel_specgram = torch.rand(
            n_batch,
            n_mels,
            max_mel_specgram_length,
            dtype=self.dtype,
            device=self.device,
        )
        mel_specgram_lengths = max_mel_specgram_length * torch.ones((n_batch,), dtype=torch.int32, device=self.device)
        return text, text_lengths, mel_specgram, mel_specgram_lengths

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    @skipIfPy310
    def test_tacotron2_torchscript_consistency(self, n_batch):
        r"""Validate the torchscript consistency of a Tacotron2."""
        n_mels = 80
        max_mel_specgram_length = 300
        max_text_length = 100

        model = _get_tacotron2_model(n_mels).to(self.device).eval()
        inputs = self._get_inputs(n_mels, n_batch, max_mel_specgram_length, max_text_length)

        self._assert_torchscript_consistency(model, inputs)

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    def test_tacotron2_output_shape(self, n_batch):
        r"""Feed tensors with specific shape to Tacotron2 and validate
        that it outputs with a tensor with expected shape.
        """
        n_mels = 80
        max_mel_specgram_length = 300
        max_text_length = 100

        model = _get_tacotron2_model(n_mels).to(self.device).eval()
        inputs = self._get_inputs(n_mels, n_batch, max_mel_specgram_length, max_text_length)
        mel_out, mel_out_postnet, gate_outputs, alignments = model(*inputs)

        assert mel_out.size() == (n_batch, n_mels, max_mel_specgram_length)
        assert mel_out_postnet.size() == (n_batch, n_mels, max_mel_specgram_length)
        assert gate_outputs.size() == (n_batch, max_mel_specgram_length)
        assert alignments.size() == (n_batch, max_mel_specgram_length, max_text_length)

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    def test_tacotron2_backward(self, n_batch):
        r"""Make sure calling the backward function on Tacotron2's outputs does
        not error out. Following:
        https://github.com/pytorch/vision/blob/23b8760374a5aaed53c6e5fc83a7e83dbe3b85df/test/test_models.py#L255
        """
        n_mels = 80
        max_mel_specgram_length = 300
        max_text_length = 100

        model = _get_tacotron2_model(n_mels).to(self.device)
        inputs = self._get_inputs(n_mels, n_batch, max_mel_specgram_length, max_text_length)
        mel_out, mel_out_postnet, gate_outputs, _ = model(*inputs)

        mel_out.sum().backward(retain_graph=True)
        mel_out_postnet.sum().backward(retain_graph=True)
        gate_outputs.sum().backward()

    def _get_inference_inputs(self, n_batch: int, max_text_length: int):
        text = torch.randint(0, 148, (n_batch, max_text_length), dtype=torch.int32, device=self.device)
        text_lengths = max_text_length * torch.ones((n_batch,), dtype=torch.int32, device=self.device)
        return text, text_lengths

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    @skipIfPy310
    def test_tacotron2_inference_torchscript_consistency(self, n_batch):
        r"""Validate the torchscript consistency of Tacotron2 inference function."""
        n_mels = 40
        max_text_length = 100
        decoder_max_step = 200  # make inference more efficient
        gate_threshold = 0.51  # if set to 0.5, the model will only run one step

        model = (
            _get_tacotron2_model(n_mels, decoder_max_step=decoder_max_step, gate_threshold=gate_threshold)
            .to(self.device)
            .eval()
        )
        inputs = self._get_inference_inputs(n_batch, max_text_length)

        model_wrapper = Tacotron2InferenceWrapper(model)

        self._assert_torchscript_consistency(model_wrapper, inputs)

    @parameterized.expand(
        [
            (1,),
            (16,),
        ]
    )
    def test_tacotron2_inference_output_shape(self, n_batch):
        r"""Feed tensors with specific shape to Tacotron2 inference function and validate
        that it outputs with a tensor with expected shape.
        """
        n_mels = 40
        max_text_length = 100
        decoder_max_step = 200  # make inference more efficient
        gate_threshold = 0.51  # if set to 0.5, the model will only run one step

        model = (
            _get_tacotron2_model(n_mels, decoder_max_step=decoder_max_step, gate_threshold=gate_threshold)
            .to(self.device)
            .eval()
        )
        inputs = self._get_inference_inputs(n_batch, max_text_length)

        mel_out, mel_specgram_lengths, alignments = model.infer(*inputs)

        # There is no guarantee on exactly what max_mel_specgram_length should be
        # We only know that it should be smaller than model.decoder.decoder_max_step
        assert len(mel_out.size()) == 3
        assert mel_out.size()[:2] == (
            n_batch,
            n_mels,
        )
        assert mel_out.size()[2] == mel_specgram_lengths.max().item()
        assert len(mel_specgram_lengths.size()) == 1
        assert mel_specgram_lengths.size()[0] == n_batch
        assert mel_specgram_lengths.max().item() <= model.decoder.decoder_max_step
        assert len(alignments.size()) == 3
        assert alignments.size()[0] == n_batch
        assert alignments.size()[1] == mel_specgram_lengths.max().item()
        assert alignments.size()[2] == max_text_length
