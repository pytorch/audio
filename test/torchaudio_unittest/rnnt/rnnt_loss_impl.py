import numpy as np
from torchaudio.prototype.rnnt_loss import RNNTLoss

from .utils import (
    compute_with_numpy_transducer,
    compute_with_pytorch_transducer,
    get_B1_T10_U3_D4_data,
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
        for reuse_logits_for_grads in [False, True]:
            with self.subTest(reuse_logits_for_grads=reuse_logits_for_grads):
                costs, gradients = compute_with_pytorch_transducer(
                    data=data, reuse_logits_for_grads=reuse_logits_for_grads
                )
                np.testing.assert_allclose(costs, ref_costs, atol=atol, rtol=rtol)
                self.assertEqual(logits_shape, gradients.shape)
                if not np.allclose(gradients, ref_gradients, atol=atol, rtol=rtol):
                    for b in range(len(gradients)):
                        T = data["logit_lengths"][b]
                        U = data["target_lengths"][b]
                        for t in range(gradients.shape[1]):
                            for u in range(gradients.shape[2]):
                                np.testing.assert_allclose(
                                    gradients[b, t, u],
                                    ref_gradients[b, t, u],
                                    atol=atol,
                                    rtol=rtol,
                                    err_msg=f"failed on b={b}, t={t}/T={T}, u={u}/U={U}",
                                )

    def test_basic_backward(self):
        rnnt_loss = RNNTLoss()
        logits, targets, logit_lengths, target_lengths = get_data_basic(self.device)
        loss = rnnt_loss(logits, targets, logit_lengths, target_lengths)
        loss.backward()

    def test_costs_and_gradients_B1_T2_U3_D5_fp32(self):
        data, ref_costs, ref_gradients = get_numpy_data_B1_T2_U3_D5(
            dtype=np.float32
        )
        data = numpy_to_torch(data=data, device=self.device, requires_grad=True)
        self._test_costs_and_gradients(
            data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
        )

    def test_costs_and_gradients_B1_T2_U3_D5_fp16(self):
        data, ref_costs, ref_gradients = get_numpy_data_B1_T2_U3_D5(
            dtype=np.float16
        )
        data = numpy_to_torch(data=data, device=self.device, requires_grad=True)
        self._test_costs_and_gradients(
            data=data,
            ref_costs=ref_costs,
            ref_gradients=ref_gradients,
            atol=1e-3,
            rtol=1e-2,
        )

    def test_costs_and_gradients_B2_T4_U3_D3_fp32(self):
        data, ref_costs, ref_gradients = get_numpy_data_B2_T4_U3_D3(
            dtype=np.float32
        )
        data = numpy_to_torch(data=data, device=self.device, requires_grad=True)
        self._test_costs_and_gradients(
            data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
        )

    def test_costs_and_gradients_B2_T4_U3_D3_fp16(self):
        data, ref_costs, ref_gradients = get_numpy_data_B2_T4_U3_D3(
            dtype=np.float16
        )
        data = numpy_to_torch(data=data, device=self.device, requires_grad=True)
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
            data = get_numpy_random_data(dtype=np.float32, seed=(seed + i))
            data = numpy_to_torch(data=data, device=self.device, requires_grad=True)
            ref_costs, ref_gradients = compute_with_numpy_transducer(data=data)
            self._test_costs_and_gradients(
                data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
            )

    def test_rnnt_nonfused_log_softmax(self):
        for random in [False, True]:
            data = get_B1_T10_U3_D4_data(
                random=random,
            )
            data = numpy_to_torch(
                data=data, device=self.device, requires_grad=True
            )
            data["fused_log_softmax"] = False
            ref_costs, ref_gradients = compute_with_numpy_transducer(
                data=data
            )
            self._test_costs_and_gradients(
                data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
            )
