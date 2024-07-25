import torch
from pipeline_tacotron2.loss import Tacotron2Loss
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import TestBaseMixin, torch_script


class Tacotron2LossInputMixin(TestBaseMixin):
    def _get_inputs(self, n_mel=80, n_batch=16, max_mel_specgram_length=300):
        mel_specgram = torch.rand(n_batch, n_mel, max_mel_specgram_length, dtype=self.dtype, device=self.device)
        mel_specgram_postnet = torch.rand(n_batch, n_mel, max_mel_specgram_length, dtype=self.dtype, device=self.device)
        gate_out = torch.rand(n_batch, dtype=self.dtype, device=self.device)
        truth_mel_specgram = torch.rand(n_batch, n_mel, max_mel_specgram_length, dtype=self.dtype, device=self.device)
        truth_gate_out = torch.rand(n_batch, dtype=self.dtype, device=self.device)

        truth_mel_specgram.requires_grad = False
        truth_gate_out.requires_grad = False

        return (
            mel_specgram,
            mel_specgram_postnet,
            gate_out,
            truth_mel_specgram,
            truth_gate_out,
        )


class Tacotron2LossShapeTests(Tacotron2LossInputMixin):
    def test_tacotron2_loss_shape(self):
        """Validate the output shape of Tacotron2Loss."""
        n_batch = 16

        (
            mel_specgram,
            mel_specgram_postnet,
            gate_out,
            truth_mel_specgram,
            truth_gate_out,
        ) = self._get_inputs(n_batch=n_batch)

        mel_loss, mel_postnet_loss, gate_loss = Tacotron2Loss()(
            (mel_specgram, mel_specgram_postnet, gate_out), (truth_mel_specgram, truth_gate_out)
        )

        self.assertEqual(mel_loss.size(), torch.Size([]))
        self.assertEqual(mel_postnet_loss.size(), torch.Size([]))
        self.assertEqual(gate_loss.size(), torch.Size([]))


class Tacotron2LossTorchscriptTests(Tacotron2LossInputMixin):
    def _assert_torchscript_consistency(self, fn, tensors):
        ts_func = torch_script(fn)

        output = fn(tensors[:3], tensors[3:])
        ts_output = ts_func(tensors[:3], tensors[3:])

        self.assertEqual(ts_output, output)

    def test_tacotron2_loss_torchscript_consistency(self):
        """Validate the torchscript consistency of Tacotron2Loss."""

        loss_fn = Tacotron2Loss()
        self._assert_torchscript_consistency(loss_fn, self._get_inputs())


class Tacotron2LossGradcheckTests(Tacotron2LossInputMixin):
    def test_tacotron2_loss_gradcheck(self):
        """Performing gradient check on Tacotron2Loss."""
        (
            mel_specgram,
            mel_specgram_postnet,
            gate_out,
            truth_mel_specgram,
            truth_gate_out,
        ) = self._get_inputs()

        mel_specgram.requires_grad_(True)
        mel_specgram_postnet.requires_grad_(True)
        gate_out.requires_grad_(True)

        def _fn(mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out):
            loss_fn = Tacotron2Loss()
            return loss_fn(
                (mel_specgram, mel_specgram_postnet, gate_out),
                (truth_mel_specgram, truth_gate_out),
            )

        gradcheck(
            _fn,
            (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out),
            fast_mode=True,
        )
        gradgradcheck(
            _fn,
            (mel_specgram, mel_specgram_postnet, gate_out, truth_mel_specgram, truth_gate_out),
            fast_mode=True,
        )
