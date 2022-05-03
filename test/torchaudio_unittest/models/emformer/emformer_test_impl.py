from abc import ABC, abstractmethod

import torch
from torchaudio.models import Emformer
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class EmformerTestMixin(ABC):
    @abstractmethod
    def gen_model(self, input_dim, right_context_length):
        pass

    @abstractmethod
    def gen_inputs(self, input_dim, batch_size, num_frames, right_context_length):
        pass

    def setUp(self):
        super().setUp()
        torch.random.manual_seed(29)

    def test_torchscript_consistency_forward(self):
        r"""Verify that scripting Emformer does not change the behavior of method `forward`."""
        input_dim = 128
        batch_size = 10
        num_frames = 400
        right_context_length = 1

        emformer = self.gen_model(input_dim, right_context_length)
        input, lengths = self.gen_inputs(input_dim, batch_size, num_frames, right_context_length)
        scripted = torch_script(emformer)

        ref_out, ref_len = emformer(input, lengths)
        scripted_out, scripted_len = scripted(input, lengths)

        self.assertEqual(ref_out, scripted_out)
        self.assertEqual(ref_len, scripted_len)

    def test_torchscript_consistency_infer(self):
        r"""Verify that scripting Emformer does not change the behavior of method `infer`."""
        input_dim = 128
        batch_size = 10
        num_frames = 5
        right_context_length = 1

        emformer = self.gen_model(input_dim, right_context_length).eval()
        scripted = torch_script(emformer).eval()

        ref_state, scripted_state = None, None
        for _ in range(3):
            input, lengths = self.gen_inputs(input_dim, batch_size, num_frames, right_context_length)
            ref_out, ref_len, ref_state = emformer.infer(input, lengths, ref_state)
            scripted_out, scripted_len, scripted_state = scripted.infer(input, lengths, scripted_state)
            self.assertEqual(ref_out, scripted_out)
            self.assertEqual(ref_len, scripted_len)
            self.assertEqual(ref_state, scripted_state)

    def test_output_shape_forward(self):
        r"""Check that method `forward` produces correctly-shaped outputs."""
        input_dim = 128
        batch_size = 10
        num_frames = 123
        right_context_length = 9

        emformer = self.gen_model(input_dim, right_context_length)
        input, lengths = self.gen_inputs(input_dim, batch_size, num_frames, right_context_length)

        output, output_lengths = emformer(input, lengths)

        self.assertEqual((batch_size, num_frames - right_context_length, input_dim), output.shape)
        self.assertEqual((batch_size,), output_lengths.shape)

    def test_output_shape_infer(self):
        r"""Check that method `infer` produces correctly-shaped outputs."""
        input_dim = 256
        batch_size = 5
        num_frames = 6
        right_context_length = 2

        emformer = self.gen_model(input_dim, right_context_length).eval()

        state = None
        for _ in range(3):
            input, lengths = self.gen_inputs(input_dim, batch_size, num_frames, right_context_length)
            output, output_lengths, state = emformer.infer(input, lengths, state)
            self.assertEqual((batch_size, num_frames - right_context_length, input_dim), output.shape)
            self.assertEqual((batch_size,), output_lengths.shape)

    def test_output_lengths_forward(self):
        r"""Check that method `forward` returns input `lengths` unmodified."""
        input_dim = 88
        batch_size = 13
        num_frames = 123
        right_context_length = 2

        emformer = self.gen_model(input_dim, right_context_length)
        input, lengths = self.gen_inputs(input_dim, batch_size, num_frames, right_context_length)
        _, output_lengths = emformer(input, lengths)
        self.assertEqual(lengths, output_lengths)

    def test_output_lengths_infer(self):
        r"""Check that method `infer` returns input `lengths` with right context length subtracted."""
        input_dim = 88
        batch_size = 13
        num_frames = 6
        right_context_length = 2

        emformer = self.gen_model(input_dim, right_context_length).eval()
        input, lengths = self.gen_inputs(input_dim, batch_size, num_frames, right_context_length)
        _, output_lengths, _ = emformer.infer(input, lengths)
        self.assertEqual(torch.clamp(lengths - right_context_length, min=0), output_lengths)


class EmformerTestImpl(EmformerTestMixin, TestBaseMixin):
    def gen_model(self, input_dim, right_context_length):
        emformer = Emformer(
            input_dim,
            8,
            256,
            3,
            4,
            left_context_length=30,
            right_context_length=right_context_length,
            max_memory_size=1,
        ).to(device=self.device, dtype=self.dtype)
        return emformer

    def gen_inputs(self, input_dim, batch_size, num_frames, right_context_length):
        input = torch.rand(batch_size, num_frames, input_dim).to(device=self.device, dtype=self.dtype)
        lengths = torch.randint(1, num_frames - right_context_length, (batch_size,)).to(
            device=self.device, dtype=self.dtype
        )
        return input, lengths
