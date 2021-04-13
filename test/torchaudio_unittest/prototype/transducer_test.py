import unittest

import numpy as np
import torch
from torchaudio.prototype import RNNTLoss

from .numpy_transducer import NumpyTransducerLoss
from torchaudio_unittest.common_utils import TorchaudioTestCase


def compute_with_pytorch_transducer(data, reuse_logits_for_grads=False):
    costs = RNNTLoss(
        blank=data["blank"],
        reuse_logits_for_grads=reuse_logits_for_grads,
    )(
        logits=data["logits"],
        src_lengths=data["src_lengths"],
        tgt_lengths=data["tgt_lengths"],
        targets=data["targets"],
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu().data.numpy()
    gradients = data["logits"].saved_grad.cpu().data.numpy()
    return costs, gradients


def skipIfNoTransducer(test_item):
    try:
        torch.ops.torchaudio.compute_transducer_loss
        return test_item
    except RuntimeError:
        return unittest.skip("torchaudio C++ extension is not compiled with RNN transducer loss")(test_item)


class TransducerTester:
    @staticmethod
    def _get_data_basic(device):
        # Example provided
        # in 6f73a2513dc784c59eec153a45f40bc528355b18
        # of https://github.com/HawkAaron/warp-transducer

        acts = torch.tensor(
            [
                [
                    [
                        [0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.6, 0.1, 0.1],
                        [0.1, 0.1, 0.2, 0.8, 0.1],
                    ],
                    [
                        [0.1, 0.6, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.2, 0.1, 0.1],
                        [0.7, 0.1, 0.2, 0.1, 0.1],
                    ],
                ]
            ],
            dtype=torch.float,
        )
        labels = torch.tensor([[1, 2]], dtype=torch.int)
        act_length = torch.tensor([2], dtype=torch.int)
        label_length = torch.tensor([2], dtype=torch.int)

        acts = acts.to(device)
        labels = labels.to(device)
        act_length = act_length.to(device)
        label_length = label_length.to(device)

        acts.requires_grad_(True)

        return acts, labels, act_length, label_length

    @staticmethod
    def _get_numpy_data_B2_T4_U3_D3(dtype=np.float32):
        logits = np.array(
            [
                0.065357,
                0.787530,
                0.081592,
                0.529716,
                0.750675,
                0.754135,
                0.609764,
                0.868140,
                0.622532,
                0.668522,
                0.858039,
                0.164539,
                0.989780,
                0.944298,
                0.603168,
                0.946783,
                0.666203,
                0.286882,
                0.094184,
                0.366674,
                0.736168,
                0.166680,
                0.714154,
                0.399400,
                0.535982,
                0.291821,
                0.612642,
                0.324241,
                0.800764,
                0.524106,
                0.779195,
                0.183314,
                0.113745,
                0.240222,
                0.339470,
                0.134160,
                0.505562,
                0.051597,
                0.640290,
                0.430733,
                0.829473,
                0.177467,
                0.320700,
                0.042883,
                0.302803,
                0.675178,
                0.569537,
                0.558474,
                0.083132,
                0.060165,
                0.107958,
                0.748615,
                0.943918,
                0.486356,
                0.418199,
                0.652408,
                0.024243,
                0.134582,
                0.366342,
                0.295830,
                0.923670,
                0.689929,
                0.741898,
                0.250005,
                0.603430,
                0.987289,
                0.592606,
                0.884672,
                0.543450,
                0.660770,
                0.377128,
                0.358021,
            ],
            dtype=dtype,
        ).reshape(2, 4, 3, 3)

        targets = np.array([[1, 2], [1, 1]], dtype=np.int32)
        src_lengths = np.array([4, 4], dtype=np.int32)
        tgt_lengths = np.array([2, 2], dtype=np.int32)

        blank = 0

        ref_costs = np.array([4.2806528590890736, 3.9384369822503591], dtype=dtype)

        ref_gradients = np.array(
            [
                -0.186844,
                -0.062555,
                0.249399,
                -0.203377,
                0.202399,
                0.000977,
                -0.141016,
                0.079123,
                0.061893,
                -0.011552,
                -0.081280,
                0.092832,
                -0.154257,
                0.229433,
                -0.075176,
                -0.246593,
                0.146405,
                0.100188,
                -0.012918,
                -0.061593,
                0.074512,
                -0.055986,
                0.219831,
                -0.163845,
                -0.497627,
                0.209240,
                0.288387,
                0.013605,
                -0.030220,
                0.016615,
                0.113925,
                0.062781,
                -0.176706,
                -0.667078,
                0.367659,
                0.299419,
                -0.356344,
                -0.055347,
                0.411691,
                -0.096922,
                0.029459,
                0.067463,
                -0.063518,
                0.027654,
                0.035863,
                -0.154499,
                -0.073942,
                0.228441,
                -0.166790,
                -0.000088,
                0.166878,
                -0.172370,
                0.105565,
                0.066804,
                0.023875,
                -0.118256,
                0.094381,
                -0.104707,
                -0.108934,
                0.213642,
                -0.369844,
                0.180118,
                0.189726,
                0.025714,
                -0.079462,
                0.053748,
                0.122328,
                -0.238789,
                0.116460,
                -0.598687,
                0.302203,
                0.296484,
            ],
            dtype=dtype,
        ).reshape(2, 4, 3, 3)

        data = {
            "logits": logits,
            "targets": targets,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "blank": blank,
        }

        return data, ref_costs, ref_gradients

    @staticmethod
    def _get_numpy_data_B1_T2_U3_D5(dtype=np.float32):
        logits = np.array(
            [
                0.1,
                0.6,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.6,
                0.1,
                0.1,
                0.1,
                0.1,
                0.2,
                0.8,
                0.1,
                0.1,
                0.6,
                0.1,
                0.1,
                0.1,
                0.1,
                0.1,
                0.2,
                0.1,
                0.1,
                0.7,
                0.1,
                0.2,
                0.1,
                0.1,
            ],
            dtype=dtype,
        ).reshape(1, 2, 3, 5)
        targets = np.array([[1, 2]], dtype=np.int32)
        src_lengths = np.array([2], dtype=np.int32)
        tgt_lengths = np.array([2], dtype=np.int32)

        blank = -1

        ref_costs = np.array([5.09566688538], dtype=dtype)
        ref_gradients = np.array(
            [
                0.17703132,
                -0.39992708,
                0.17703132,
                0.17703132,
                -0.13116692,
                0.12247062,
                0.12247062,
                -0.181684,
                0.12247062,
                -0.1857276,
                0.06269141,
                0.06269141,
                0.06928471,
                0.12624498,
                -0.32091248,
                0.05456069,
                -0.2182428,
                0.05456069,
                0.05456069,
                0.05456069,
                0.12073967,
                0.12073967,
                -0.48295838,
                0.12073967,
                0.12073967,
                0.30741188,
                0.16871123,
                0.18645471,
                0.16871123,
                -0.83128875,
            ],
            dtype=dtype,
        ).reshape(1, 2, 3, 5)

        data = {
            "logits": logits,
            "targets": targets,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "blank": blank,
        }

        return data, ref_costs, ref_gradients

    @staticmethod
    def _get_numpy_random_data(
        max_B=8, max_T=128, max_U=32, max_D=40, blank=-1, dtype=np.float32, seed=None
    ):
        if seed is not None:
            np.random.seed(seed=seed)

        if blank != -1:
            raise ValueError("blank != -1 is not supported yet.")

        B = np.random.randint(low=1, high=max_B)
        T = np.random.randint(low=5, high=max_T)
        U = np.random.randint(low=5, high=max_U)
        D = np.random.randint(low=2, high=max_D)

        src_lengths = np.random.randint(low=5, high=T + 1, size=(B,), dtype=np.int32)
        tgt_lengths = np.random.randint(low=5, high=U + 1, size=(B,), dtype=np.int32)
        max_src_length = np.max(src_lengths)
        max_tgt_length = np.max(tgt_lengths)
        targets = np.random.randint(
            low=0, high=D - 1, size=(B, max_tgt_length), dtype=np.int32
        )
        logits = np.random.random_sample(
            size=(B, max_src_length, max_tgt_length + 1, D)
        ).astype(dtype=dtype)

        return {
            "logits": logits,
            "targets": targets,
            "src_lengths": src_lengths,
            "tgt_lengths": tgt_lengths,
            "blank": blank,
        }

    @staticmethod
    def _numpy_to_torch(data, device, requires_grad=True):
        logits = torch.from_numpy(data["logits"])
        targets = torch.from_numpy(data["targets"])
        src_lengths = torch.from_numpy(data["src_lengths"])
        tgt_lengths = torch.from_numpy(data["tgt_lengths"])

        if "nbest_wers" in data:
            data["nbest_wers"] = torch.from_numpy(data["nbest_wers"]).to(device)
        if "nbest_scores" in data:
            data["nbest_scores"] = torch.from_numpy(data["nbest_scores"]).to(device)

        logits = torch.autograd.Variable(logits, requires_grad=requires_grad)
        src_lengths = torch.autograd.Variable(src_lengths)
        tgt_lengths = torch.autograd.Variable(tgt_lengths)
        targets = torch.autograd.Variable(targets)

        if device == "cpu":
            logits = logits.cpu()
        else:
            raise ValueError("unrecognized device = {}".format(device))

        def grad_hook(grad):
            logits.saved_grad = grad.clone()

        logits.register_hook(grad_hook)

        data["logits"] = logits
        data["src_lengths"] = src_lengths
        data["tgt_lengths"] = tgt_lengths
        data["targets"] = targets

        return data

    @staticmethod
    def _compute_with_numpy_transducer(data):
        costs = NumpyTransducerLoss(
            blank=data["blank"],
        )(
            logits=data["logits"],
            src_lengths=data["src_lengths"],
            tgt_lengths=data["tgt_lengths"],
            targets=data["targets"],
        )

        loss = torch.sum(costs)
        loss.backward()

        costs = costs.cpu().data.numpy()
        gradients = data["logits"].saved_grad.cpu().data.numpy()

        return costs, gradients

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
                        T = data["src_lengths"][b]
                        U = data["tgt_lengths"][b]
                        for t in range(gradients.shape[1]):
                            for u in range(gradients.shape[2]):
                                np.testing.assert_allclose(
                                    gradients[b, t, u],
                                    ref_gradients[b, t, u],
                                    atol=atol,
                                    rtol=rtol,
                                    err_msg=f"failed on b={b}, t={t}/T={T}, u={u}/U={U}",
                                )

    def test_basic_fp16_no_error(self):
        """RNNT loss should support fp16
        """
        rnnt_loss = RNNTLoss()
        acts, labels, act_length, label_length = self._get_data_basic(device=self.device)
        acts = acts.to(torch.float16)
        rnnt_loss(acts, labels, act_length, label_length)

    def test_basic_backward(self):
        rnnt_loss = RNNTLoss()
        acts, labels, act_length, label_length = self._get_data_basic(device=self.device)
        loss = rnnt_loss(acts, labels, act_length, label_length)
        loss.backward()

    def test_costs_and_gradients_B1_T2_U3_D5_fp32_cpu(self):
        data, ref_costs, ref_gradients = self._get_numpy_data_B1_T2_U3_D5(
            dtype=np.float32
        )
        data = self._numpy_to_torch(data=data, device=self.device, requires_grad=True)
        self._test_costs_and_gradients(
            data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
        )

    def test_costs_and_gradients_B2_T4_U3_D3_fp32_cpu(self):
        data, ref_costs, ref_gradients = self._get_numpy_data_B2_T4_U3_D3(
            dtype=np.float32
        )
        data = self._numpy_to_torch(data=data, device=self.device, requires_grad=True)
        self._test_costs_and_gradients(
            data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
        )

    def test_costs_and_gradients_random_data_with_numpy_fp32_cpu(self):
        seed = 777
        for i in range(5):
            data = self._get_numpy_random_data(dtype=np.float32, seed=(seed + i))
            data = self._numpy_to_torch(data=data, device=self.device, requires_grad=True)
            ref_costs, ref_gradients = self._compute_with_numpy_transducer(data=data)
            self._test_costs_and_gradients(
                data=data, ref_costs=ref_costs, ref_gradients=ref_gradients
            )


@skipIfNoTransducer
class CPUTransducerTester(TransducerTester, TorchaudioTestCase):
    device = "cpu"
