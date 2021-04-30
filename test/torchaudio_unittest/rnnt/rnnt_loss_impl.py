import numpy as np
import torch
from torchaudio.prototype.rnnt_loss import (
    RNNTLoss,
    _rnnt_loss_alphas,
    _rnnt_loss_betas,
)

from .numpy_transducer import _NumpyTransducer

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

from torchaudio_unittest.common_utils import skipIfNoCuda


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

    def test_alphas_restricted_B1_T10_U3_D4(self):
        for random in [True, False]:
            for left_buffer in [0, 2, 10]:
                for right_buffer in [0, 1, 2, 10]:
                    data = get_B1_T10_U3_D4_data(random=random)
                    logits = torch.tensor(data["logits"])
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).numpy()
                    (
                        np_gradients,
                        np_costs,
                        np_alphas,
                        np_betas,
                    ) = _NumpyTransducer.compute(
                        log_probs=log_probs,
                        logit_lengths=data["logit_lengths"],
                        target_lengths=data["target_lengths"],
                        targets=data["targets"],
                        blank=data["blank"],
                        wordpiece_ends=data["wordpiece_ends"],
                        left_buffer=left_buffer,
                        right_buffer=right_buffer,
                    )

                    data = numpy_to_torch(
                        data=data, device=self.device, requires_grad=True
                    )
                    alphas = _rnnt_loss_alphas(
                        logits=data["logits"],
                        logit_lengths=data["logit_lengths"],
                        target_lengths=data["target_lengths"],
                        targets=data["targets"],
                        wordpiece_ends=data.get("wordpiece_ends", None),
                        blank=data["blank"],
                        left_buffer=left_buffer,
                        right_buffer=right_buffer,
                    )
                    np.testing.assert_allclose(
                        alphas.cpu().data.numpy(), np_alphas, atol=1e-6, rtol=1e-2
                    )

    def test_betas_restricted_B1_T10_U3_D4(self):
        for random in [True, False]:
            for left_buffer in [0, 2, 10]:
                for right_buffer in [0, 1, 2, 10]:
                    data = get_B1_T10_U3_D4_data(random=random)
                    logits = torch.tensor(data["logits"])
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).numpy()
                    (
                        np_gradients,
                        np_costs,
                        np_alphas,
                        np_betas,
                    ) = _NumpyTransducer.compute(
                        log_probs=log_probs,
                        logit_lengths=data["logit_lengths"],
                        target_lengths=data["target_lengths"],
                        targets=data["targets"],
                        blank=data["blank"],
                        wordpiece_ends=data["wordpiece_ends"],
                        left_buffer=left_buffer,
                        right_buffer=right_buffer,
                    )

                    data = numpy_to_torch(
                        data=data, device=self.device, requires_grad=True
                    )
                    betas = _rnnt_loss_betas(
                        logits=data["logits"],
                        logit_lengths=data["logit_lengths"],
                        target_lengths=data["target_lengths"],
                        targets=data["targets"],
                        wordpiece_ends=data.get("wordpiece_ends", None),
                        blank=data["blank"],
                        left_buffer=left_buffer,
                        right_buffer=right_buffer,
                    )
                    np.testing.assert_allclose(
                        betas.cpu().data.numpy(), np_betas, atol=1e-6, rtol=1e-2
                    )

    def test_rnnt_restricted_B1_T10_U3_D4(self):
        for left_buffer in [0, 1, 10]:
            for right_buffer in [0, 1, 2, 5, 10]:
                data = get_B1_T10_U3_D4_data(random=False)
                data = numpy_to_torch(
                    data=data, device=self.device, requires_grad=True
                )
                data["left_buffer"] = left_buffer
                data["right_buffer"] = right_buffer
                ref_costs, ref_gradients = compute_with_numpy_transducer(
                    data=data
                )
                self._test_costs_and_gradients(
                    data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                )

    def test_rnnt_restricted_B2_T4_U3_D3(self):
        # Note - this test just ensures that the numpy and c++ implementations match.
        # Probably a hand constructed test for gradients will be more thorough
        data, _, _ = get_numpy_data_B2_T4_U3_D3(dtype=np.float32)
        data = numpy_to_torch(data=data, device=self.device, requires_grad=True)
        data["wordpiece_ends"] = torch.tensor([[0, 1, 2], [0, 1, 2]]).int()

        for left_buffer in [0]:
            for right_buffer in [0, 1, 2]:
                data["left_buffer"] = left_buffer
                data["right_buffer"] = right_buffer
                ref_costs, ref_gradients = compute_with_numpy_transducer(
                    data=data
                )

                self._test_costs_and_gradients(
                    data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                )

    def test_restricted_parity_with_unrestricted_B1_T10_U3_D4(self):
        for random in [False, True]:
            data = get_B1_T10_U3_D4_data(random=random)
            data = numpy_to_torch(data=data, device=self.device, requires_grad=True)
            wordpiece_ends = data["wordpiece_ends"]
            del data["wordpiece_ends"]
            ref_costs, ref_gradients = compute_with_pytorch_transducer(data=data)
            data["wordpiece_ends"] = wordpiece_ends
            data["left_buffer"] = 100
            data["right_buffer"] = 100
            wp_costs, wp_gradients = compute_with_pytorch_transducer(data=data)
            np.testing.assert_allclose(ref_costs, wp_costs, atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(
                ref_gradients, wp_gradients, atol=1e-2, rtol=1e-2
            )

    def test_restricted_parity_with_unrestricted_alpha_B1_T10_U3_D4(self):
        for random in [False, True]:
            data = get_B1_T10_U3_D4_data(random=random)
            data = numpy_to_torch(data=data, device=self.device, requires_grad=True)

            ref_alphas = _rnnt_loss_alphas(
                blank=data["blank"],
                logits=data["logits"],
                logit_lengths=data["logit_lengths"],
                target_lengths=data["target_lengths"],
                targets=data["targets"],
            )

            wp_alphas = _rnnt_loss_alphas(
                blank=data["blank"],
                left_buffer=100,
                right_buffer=100,
                logits=data["logits"],
                logit_lengths=data["logit_lengths"],
                target_lengths=data["target_lengths"],
                targets=data["targets"],
                wordpiece_ends=data["wordpiece_ends"],
            )
            np.testing.assert_allclose(
                ref_alphas.cpu().data.numpy(),
                wp_alphas.cpu().data.numpy(),
                atol=1e-2,
                rtol=1e-2,
            )

    def test_restricted_parity_with_unrestricted_beta_B1_T10_U3_D4(self):
        for random in [False, True]:
            data = get_B1_T10_U3_D4_data(random=random)
            data = numpy_to_torch(data=data, device=self.device, requires_grad=True)

            ref_betas = _rnnt_loss_betas(
                blank=data["blank"],
                logits=data["logits"],
                logit_lengths=data["logit_lengths"],
                target_lengths=data["target_lengths"],
                targets=data["targets"],
            )
            wp_betas = _rnnt_loss_betas(
                blank=data["blank"],
                left_buffer=100,
                right_buffer=100,
                logits=data["logits"],
                logit_lengths=data["logit_lengths"],
                target_lengths=data["target_lengths"],
                targets=data["targets"],
                wordpiece_ends=data["wordpiece_ends"],
            )
            np.testing.assert_allclose(
                ref_betas.cpu().data.numpy(),
                wp_betas.cpu().data.numpy(),
                atol=1e-2,
                rtol=1e-2,
            )


@skipIfNoCuda
class RNNTLossCUDAOnly:
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

    def test_nan_logits(self):
        for reuse_logits_for_grads in [False, True]:
            data = get_B1_T10_U3_D4_data(
                random=True, left_buffer=10, right_buffer=10, nan=True
            )
            data = numpy_to_torch(
                data=data, device=self.device, requires_grad=True
            )
            data["left_buffer"] = 10
            data["right_buffer"] = 10
            costs, gradients = compute_with_pytorch_transducer(
                data=data, reuse_logits_for_grads=reuse_logits_for_grads
            )
            self.assertTrue(np.all(costs == 0))
            self.assertTrue(np.all(gradients == 0))

    def test_rnnt_restricted_betas_with_random_data(self):
        for u in range(50, 60):
            data = get_numpy_random_data(max_B=5, max_T=499, max_U=u, max_D=4096)
            data = numpy_to_torch(
                data=data, device=self.device, requires_grad=True
            )
            targets = data["targets"]
            target_lengths = data["target_lengths"]
            B, max_U = targets.size()
            wordpiece_ends = np.zeros((B, max_U + 1), dtype=np.int32)
            for b in range(B):
                u = target_lengths[b]
                wordpiece_ends[b, : u + 1] = np.array([0] + list(range(u)), dtype=np.int32)
            wordpiece_ends = torch.from_numpy(wordpiece_ends)
            _ = _rnnt_loss_betas(
                logits=data["logits"],
                logit_lengths=data["logit_lengths"],
                target_lengths=data["target_lengths"],
                targets=data["targets"],
                wordpiece_ends=wordpiece_ends,
                blank=4095,
            )

    def test_rnnt_nonfused_log_softmax(self):
        for random in [False, True]:
            for left_buffer in [1, 2, 10]:
                for right_buffer in [1, 2, 5, 10]:
                    data = get_B1_T10_U3_D4_data(
                        random=random,
                        left_buffer=left_buffer,
                        right_buffer=right_buffer,
                    )
                    data = numpy_to_torch(
                        data=data, device=self.device, requires_grad=True
                    )
                    data["left_buffer"] = left_buffer
                    data["right_buffer"] = right_buffer
                    data["fused_log_softmax"] = False
                    ref_costs, ref_gradients = compute_with_numpy_transducer(
                        data=data
                    )
                    self._test_costs_and_gradients(
                        data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
                    )
