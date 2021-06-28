import torch
from torchaudio.prototype.rnnt_loss import RNNTLoss

from .utils import (
    compute_with_numpy_transducer,
    compute_with_pytorch_transducer,
    get_data_basic,
    get_numpy_data_B1_T2_U3_D5,
    get_numpy_data_B2_T4_U3_D3,
    get_numpy_random_data,
    numpy_to_torch,
)


class RNNTLossTest:
    def _test_costs_and_gradients(
        self, data, ref_costs, ref_gradients, atol=1e-6, rtol=1e-2
    ):
        logits_shape = data["logits"].shape
        costs, gradients = compute_with_pytorch_transducer(data=data)
        self.assertEqual(costs, ref_costs, atol=atol, rtol=rtol)
        self.assertEqual(logits_shape, gradients.shape)
        self.assertEqual(gradients, ref_gradients, atol=atol, rtol=rtol)

    def test_basic_backward(self):
        rnnt_loss = RNNTLoss()
        logits, targets, logit_lengths, target_lengths = get_basic_data(self.device)
        loss = rnnt_loss(logits, targets, logit_lengths, target_lengths)
        loss.backward()

    def test_costs_and_gradients_B1_T2_U3_D5_fp32(self):
        data, ref_costs, ref_gradients = get_B1_T2_U3_D5_data(
            dtype=torch.float32,
            device=self.device,
        )
        self._test_costs_and_gradients(
            data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
        )

    def test_costs_and_gradients_B1_T2_U3_D5_fp16(self):
        data, ref_costs, ref_gradients = get_B1_T2_U3_D5_data(
            dtype=torch.float16,
            device=self.device,
        )
        self._test_costs_and_gradients(
            data=data,
            ref_costs=ref_costs,
            ref_gradients=ref_gradients,
            atol=1e-3,
            rtol=1e-2,
        )

    def test_costs_and_gradients_B2_T4_U3_D3_fp32(self):
        data, ref_costs, ref_gradients = get_B2_T4_U3_D3_data(
            dtype=torch.float32,
            device=self.device,
        )
        self._test_costs_and_gradients(
            data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
        )

    def test_costs_and_gradients_B2_T4_U3_D3_fp16(self):
        data, ref_costs, ref_gradients = get_B2_T4_U3_D3_data(
            dtype=torch.float16,
            device=self.device,
        )
        self._test_costs_and_gradients(
            data=data,
            ref_costs=ref_costs,
            ref_gradients=ref_gradients,
            atol=1e-3,
            rtol=1e-2,
        )

    def test_costs_and_gradients_random_data_with_numpy_fp32(self):
        seed = 777
        for i in range(5):
            data = get_random_data(dtype=torch.float32, device=self.device, seed=(seed + i))
            ref_costs, ref_gradients = compute_with_numpy_transducer(data=data)
            self._test_costs_and_gradients(
                data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
            )
