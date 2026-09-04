import torch
from torchaudio.models import Conformer
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class ConformerTestImpl(TestBaseMixin):
    def _gen_model(self, **kwargs):
        conformer = (
            Conformer(
                input_dim=80,
                num_heads=4,
                ffn_dim=128,
                num_layers=4,
                depthwise_conv_kernel_size=31,
                dropout=0.1,
                **kwargs,
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

    def test_padding_invariance(self):
        r"""Verify that with ``conv_mask_padding=True`` and ``conv_norm_type="layer_norm"``,
        the output for a sequence does not depend on the amount of padding in its batch."""
        input_dim = 80
        batch_size = 10
        num_frames = 400

        conformer = self._gen_model(conv_mask_padding=True, conv_norm_type="layer_norm")
        input, lengths = self._gen_inputs(input_dim, batch_size, num_frames)

        batched_out, _ = conformer(input, lengths)

        for i in range(batch_size):
            length = int(lengths[i])
            single_out, _ = conformer(input[i : i + 1, :length], lengths[i : i + 1])
            self.assertEqual(batched_out[i, :length], single_out[0])

    def test_invalid_norm_configs(self):
        r"""Verify that combining ``use_group_norm`` with ``conv_norm_type`` is rejected."""
        with self.assertRaises(ValueError):
            self._gen_model(use_group_norm=True, conv_norm_type="layer_norm")
