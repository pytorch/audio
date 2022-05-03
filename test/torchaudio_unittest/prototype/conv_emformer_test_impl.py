import torch
from torchaudio.prototype.models.conv_emformer import ConvEmformer
from torchaudio_unittest.common_utils import TestBaseMixin
from torchaudio_unittest.models.emformer.emformer_test_impl import EmformerTestMixin


class ConvEmformerTestImpl(EmformerTestMixin, TestBaseMixin):
    def gen_model(self, input_dim, right_context_length):
        emformer = ConvEmformer(
            input_dim,
            8,
            256,
            3,
            4,
            12,
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
