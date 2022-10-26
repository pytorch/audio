import random
import unittest

import numpy as np
import torch
from torchaudio.functional import rnnt_loss


CPU_DEVICE = torch.device("cpu")


class _NumpyTransducer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        logit_lengths,
        target_lengths,
        targets,
        blank=-1,
    ):
        device = log_probs.device
        log_probs = log_probs.cpu().data.numpy()
        logit_lengths = logit_lengths.cpu().data.numpy()
        target_lengths = target_lengths.cpu().data.numpy()
        targets = targets.cpu().data.numpy()

        gradients, costs, _, _ = __class__.compute(
            log_probs=log_probs,
            logit_lengths=logit_lengths,
            target_lengths=target_lengths,
            targets=targets,
            blank=blank,
        )

        costs = torch.FloatTensor(costs).to(device=device)
        gradients = torch.FloatTensor(gradients).to(device=device)
        ctx.grads = torch.autograd.Variable(gradients)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul(grad_output), None, None, None, None, None, None, None, None

    @staticmethod
    def compute_alpha_one_sequence(log_probs, targets, blank=-1):
        max_T, max_U, D = log_probs.shape
        alpha = np.zeros((max_T, max_U), dtype=np.float32)
        for t in range(1, max_T):
            alpha[t, 0] = alpha[t - 1, 0] + log_probs[t - 1, 0, blank]

        for u in range(1, max_U):
            alpha[0, u] = alpha[0, u - 1] + log_probs[0, u - 1, targets[u - 1]]

        for t in range(1, max_T):
            for u in range(1, max_U):
                skip = alpha[t - 1, u] + log_probs[t - 1, u, blank]
                emit = alpha[t, u - 1] + log_probs[t, u - 1, targets[u - 1]]
                alpha[t, u] = np.logaddexp(skip, emit)

        cost = -(alpha[-1, -1] + log_probs[-1, -1, blank])
        return alpha, cost

    @staticmethod
    def compute_beta_one_sequence(log_probs, targets, blank=-1):
        max_T, max_U, D = log_probs.shape
        beta = np.zeros((max_T, max_U), dtype=np.float32)
        beta[-1, -1] = log_probs[-1, -1, blank]

        for t in reversed(range(max_T - 1)):
            beta[t, -1] = beta[t + 1, -1] + log_probs[t, -1, blank]

        for u in reversed(range(max_U - 1)):
            beta[-1, u] = beta[-1, u + 1] + log_probs[-1, u, targets[u]]

        for t in reversed(range(max_T - 1)):
            for u in reversed(range(max_U - 1)):
                skip = beta[t + 1, u] + log_probs[t, u, blank]
                emit = beta[t, u + 1] + log_probs[t, u, targets[u]]
                beta[t, u] = np.logaddexp(skip, emit)

        cost = -beta[0, 0]
        return beta, cost

    @staticmethod
    def compute_gradients_one_sequence(log_probs, alpha, beta, targets, blank=-1):
        max_T, max_U, D = log_probs.shape
        gradients = np.full(log_probs.shape, float("-inf"))
        cost = -beta[0, 0]

        gradients[-1, -1, blank] = alpha[-1, -1]

        gradients[:-1, :, blank] = alpha[:-1, :] + beta[1:, :]

        for u, l in enumerate(targets):
            gradients[:, u, l] = alpha[:, u] + beta[:, u + 1]

        gradients = -(np.exp(gradients + log_probs + cost))
        return gradients

    @staticmethod
    def compute(
        log_probs,
        logit_lengths,
        target_lengths,
        targets,
        blank=-1,
    ):
        gradients = np.zeros_like(log_probs)
        B_tgt, max_T, max_U, D = log_probs.shape
        B_src = logit_lengths.shape[0]

        H = int(B_tgt / B_src)

        alphas = np.zeros((B_tgt, max_T, max_U))
        betas = np.zeros((B_tgt, max_T, max_U))
        betas.fill(float("-inf"))
        alphas.fill(float("-inf"))
        costs = np.zeros(B_tgt)
        for b_tgt in range(B_tgt):
            b_src = int(b_tgt / H)
            T = int(logit_lengths[b_src])
            # NOTE: see https://arxiv.org/pdf/1211.3711.pdf Section 2.1
            U = int(target_lengths[b_tgt]) + 1

            seq_log_probs = log_probs[b_tgt, :T, :U, :]
            seq_targets = targets[b_tgt, : int(target_lengths[b_tgt])]
            alpha, alpha_cost = __class__.compute_alpha_one_sequence(
                log_probs=seq_log_probs, targets=seq_targets, blank=blank
            )

            beta, beta_cost = __class__.compute_beta_one_sequence(
                log_probs=seq_log_probs, targets=seq_targets, blank=blank
            )

            seq_gradients = __class__.compute_gradients_one_sequence(
                log_probs=seq_log_probs,
                alpha=alpha,
                beta=beta,
                targets=seq_targets,
                blank=blank,
            )
            np.testing.assert_almost_equal(alpha_cost, beta_cost, decimal=2)
            gradients[b_tgt, :T, :U, :] = seq_gradients
            costs[b_tgt] = beta_cost
            alphas[b_tgt, :T, :U] = alpha
            betas[b_tgt, :T, :U] = beta

        return gradients, costs, alphas, betas


class NumpyTransducerLoss(torch.nn.Module):
    def __init__(self, blank=-1):
        super().__init__()
        self.blank = blank

    def forward(
        self,
        logits,
        logit_lengths,
        target_lengths,
        targets,
    ):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return _NumpyTransducer.apply(
            log_probs,
            logit_lengths,
            target_lengths,
            targets,
            self.blank,
        )


def compute_with_numpy_transducer(data):
    costs = NumpyTransducerLoss(blank=data["blank"],)(
        logits=data["logits"],
        logit_lengths=data["logit_lengths"],
        target_lengths=data["target_lengths"],
        targets=data["targets"],
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu()
    gradients = data["logits"].saved_grad.cpu()
    return costs, gradients


def compute_with_pytorch_transducer(data, fused_log_softmax=True):
    costs = rnnt_loss(
        logits=data["logits"],
        logit_lengths=data["logit_lengths"],
        target_lengths=data["target_lengths"],
        targets=data["targets"],
        blank=data["blank"],
        reduction="none",
        fused_log_softmax=fused_log_softmax,
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu()
    gradients = data["logits"].saved_grad.cpu()
    return costs, gradients


def get_basic_data(device):
    # Example provided
    # in 6f73a2513dc784c59eec153a45f40bc528355b18
    # of https://github.com/HawkAaron/warp-transducer

    logits = torch.tensor(
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
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor([[1, 2]], dtype=torch.int, device=device)
    logit_lengths = torch.tensor([2], dtype=torch.int, device=device)
    target_lengths = torch.tensor([2], dtype=torch.int, device=device)

    logits.requires_grad_(True)

    return logits, targets, logit_lengths, target_lengths


def get_B1_T10_U3_D4_data(
    random=False,
    dtype=torch.float32,
    device=CPU_DEVICE,
):
    B, T, U, D = 2, 10, 3, 4

    logits = torch.rand(B, T, U, D, dtype=dtype, device=device)
    if not random:
        logits.fill_(0.1)
    logits.requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()

    logits.register_hook(grad_hook)

    data = {}
    data["logits"] = logits
    data["logit_lengths"] = torch.tensor([10, 10], dtype=torch.int32, device=device)
    data["target_lengths"] = torch.tensor([2, 2], dtype=torch.int32, device=device)
    data["targets"] = torch.tensor([[1, 2], [1, 2]], dtype=torch.int32, device=device)
    data["blank"] = 0

    return data


def get_B1_T2_U3_D5_data(dtype=torch.float32, device=CPU_DEVICE):
    logits = torch.tensor(
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
        device=device,
    ).reshape(1, 2, 3, 5)
    logits.requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()

    logits.register_hook(grad_hook)

    targets = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
    logit_lengths = torch.tensor([2], dtype=torch.int32, device=device)
    target_lengths = torch.tensor([2], dtype=torch.int32, device=device)

    blank = -1

    ref_costs = torch.tensor([5.09566688538], dtype=dtype)
    ref_gradients = torch.tensor(
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
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }

    return data, ref_costs, ref_gradients


def get_B2_T4_U3_D3_data(dtype=torch.float32, device=CPU_DEVICE):
    # Test from D21322854
    logits = torch.tensor(
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
        device=device,
    ).reshape(2, 4, 3, 3)
    logits.requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()

    logits.register_hook(grad_hook)

    targets = torch.tensor([[1, 2], [1, 1]], dtype=torch.int32, device=device)
    logit_lengths = torch.tensor([4, 4], dtype=torch.int32, device=device)
    target_lengths = torch.tensor([2, 2], dtype=torch.int32, device=device)

    blank = 0

    ref_costs = torch.tensor([4.2806528590890736, 3.9384369822503591], dtype=dtype)

    ref_gradients = torch.tensor(
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
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }

    return data, ref_costs, ref_gradients


def get_random_data(
    max_B=8,
    max_T=128,
    max_U=32,
    max_D=40,
    blank=-1,
    dtype=torch.float32,
    device=CPU_DEVICE,
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed=seed)

    if blank != -1:
        raise ValueError("blank != -1 is not supported yet.")

    random.seed(0)
    B = random.randint(1, max_B - 1)
    T = random.randint(5, max_T - 1)
    U = random.randint(5, max_U - 1)
    D = random.randint(2, max_D - 1)

    logit_lengths = torch.randint(low=5, high=T + 1, size=(B,), dtype=torch.int32, device=device)
    target_lengths = torch.randint(low=5, high=U + 1, size=(B,), dtype=torch.int32, device=device)
    max_src_length = torch.max(logit_lengths)
    max_tgt_length = torch.max(target_lengths)

    targets = torch.randint(low=0, high=D - 1, size=(B, max_tgt_length), dtype=torch.int32, device=device)
    logits = torch.rand(
        size=(B, max_src_length, max_tgt_length + 1, D),
        dtype=dtype,
        device=device,
    ).requires_grad_(True)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()

    logits.register_hook(grad_hook)

    return {
        "logits": logits,
        "targets": targets,
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }


def skipIfNoRNNT(test_item):
    try:
        torch.ops.torchaudio.rnnt_loss
        return test_item
    except RuntimeError:
        return unittest.skip("torchaudio C++ extension is not compiled with RNN transducer loss")
