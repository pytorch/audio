import unittest
import random
import torch
from torchaudio.prototype.rnnt_loss import RNNTLoss

from .numpy_transducer import NumpyTransducerLoss


def compute_with_numpy_transducer(data):
    costs = NumpyTransducerLoss(
        blank=data["blank"],
    )(
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


def compute_with_pytorch_transducer(data, reuse_logits_for_grads=False):
    costs = RNNTLoss(
        blank=data["blank"],
        fused_log_softmax=data.get("fused_log_softmax", True),
        reuse_logits_for_grads=reuse_logits_for_grads,
        reduction="none",
    )(
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
    device=torch.device("cpu"),
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


def get_B1_T2_U3_D5_data(dtype=torch.float32, device=torch.device("cpu")):
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


def get_B2_T4_U3_D3_data(dtype=torch.float32, device=torch.device("cpu")):
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
    device=torch.device("cpu"),
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed=seed)

    if blank != -1:
        raise ValueError("blank != -1 is not supported yet.")

    B = random.randint(1, max_B - 1)
    T = random.randint(5, max_T - 1)
    U = random.randint(5, max_U - 1)
    D = random.randint(2, max_D - 1)

    logit_lengths = torch.randint(low=5, high=T + 1, size=(B,), dtype=torch.int32, device=device)
    target_lengths = torch.randint(low=5, high=U + 1, size=(B,), dtype=torch.int32, device=device)
    max_src_length = torch.max(logit_lengths)
    max_tgt_length = torch.max(target_lengths)

    targets = torch.randint(
        low=0, high=D - 1, size=(B, max_tgt_length), dtype=torch.int32, device=device
    )
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
