import torch
from torchaudio.models import Conformer
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class ConformerTestImpl(TestBaseMixin):
    def _gen_model(self):
        conformer = (
            Conformer(
                input_dim=80,
                num_heads=4,
                ffn_dim=128,
                num_layers=4,
                depthwise_conv_kernel_size=31,
                dropout=0.1,
            )
            .to(device=self.device, dtype=self.dtype)
            .eval()
        )
        return conformer

    def _gen_inputs(self, input_dim, batch_size, num_frames):
        lengths = torch.randint(1, num_frames, (batch_size,)).to(device=self.device, dtype=self.dtype)
        input = torch.rand(batch_size, int(lengths.max()), input_dim).to(device=self.device, dtype=self.dtype)
        return input, lengths

    def setUp(self):
        super().setUp()

    def test_torchscript_consistency_forward(self):
        r"""Verify that scripting Conformer does not change the behavior of method `forward`."""
        input_dim = 80
        batch_size = 10
        num_frames = 400

        conformer = self._gen_model()
        input, lengths = self._gen_inputs(input_dim, batch_size, num_frames)
        scripted = torch_script(conformer)

        ref_out, ref_len = conformer(input, lengths)
        scripted_out, scripted_len = scripted(input, lengths)

        self.assertEqual(ref_out, scripted_out)
        self.assertEqual(ref_len, scripted_len)
