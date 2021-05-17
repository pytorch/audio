import torch
from torchaudio_unittest.common_utils import TempDirMixin, TestBaseMixin
from torchaudio.prototype.rnnt_loss import RNNTLoss, rnnt_loss


class RNNTLossTorchscript(TempDirMixin, TestBaseMixin):
    """Implements test for RNNT Loss that are performed for different devices"""
    def _assert_consistency(self, func, tensor, shape_only=False):
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        path = self.get_temp_path('func.zip')
        torch.jit.script(func).save(path)
        ts_func = torch.jit.load(path)

        torch.random.manual_seed(40)
        input_tensor = tensor.clone().detach().requires_grad_(True)
        output = func(input_tensor)

        torch.random.manual_seed(40)
        input_tensor = tensor.clone().detach().requires_grad_(True)
        ts_output = ts_func(input_tensor)

        self.assertEqual(ts_output, output)

    def test_rnnt_loss(self):
        def func(
            logits,
        ):
            targets = torch.tensor([[1, 2]], device=logits.device, dtype=torch.int32)
            logit_lengths = torch.tensor([2], device=logits.device, dtype=torch.int32)
            target_lengths = torch.tensor([2], device=logits.device, dtype=torch.int32)
            return rnnt_loss(logits, targets, logit_lengths, target_lengths)

        logits = torch.tensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.2, 0.8, 0.1]],
                                [[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.2, 0.1, 0.1],
                                 [0.7, 0.1, 0.2, 0.1, 0.1]]]])

        self._assert_consistency(func, logits)

    def test_RNNTLoss(self):
        func = RNNTLoss()

        logits = torch.tensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.6, 0.1, 0.1],
                                 [0.1, 0.1, 0.2, 0.8, 0.1]],
                                [[0.1, 0.6, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.2, 0.1, 0.1],
                                 [0.7, 0.1, 0.2, 0.1, 0.1]]]])
        targets = torch.tensor([[1, 2]], device=logits.device, dtype=torch.int32)
        logit_lengths = torch.tensor([2], device=logits.device, dtype=torch.int32)
        target_lengths = torch.tensor([2], device=logits.device, dtype=torch.int32)

        tensor = logits.to(device=self.device, dtype=self.dtype)

        path = self.get_temp_path('func.zip')
        torch.jit.script(func).save(path)
        ts_func = torch.jit.load(path)

        torch.random.manual_seed(40)
        input_tensor = tensor.clone().detach().requires_grad_(True)
        output = func(input_tensor, targets, logit_lengths, target_lengths)

        torch.random.manual_seed(40)
        input_tensor = tensor.clone().detach().requires_grad_(True)
        ts_output = ts_func(input_tensor, targets, logit_lengths, target_lengths)

        self.assertEqual(ts_output, output)
