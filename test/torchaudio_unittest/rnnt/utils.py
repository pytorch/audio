import unittest

import numpy as np
import torch
from torchaudio.prototype.rnnt_loss import RNNTLoss

from .numpy_transducer import (
    AlignmentRestrictionCheck,
    NumpyTransducerLoss,
)


def _get_sparse(data, dense_tensor, left_buffer, right_buffer, H=1):
    B, _, U, D = 2, 10, 3, 4
    total_valid = 0
    valid_ranges = np.zeros((B * H, U, 2), dtype=np.int32)
    cells_per_sample = np.zeros(B * H, dtype=np.int32)
    wordpiece_ends = data["wordpiece_ends"]
    for b_tgt in range(B * H):
        b_src = int(b_tgt / H)
        src_len = int(data["logit_lengths"][b_src])
        tgt_len = int(data["target_lengths"][b_tgt]) + 1
        ar_check = AlignmentRestrictionCheck(
            tgt_len, src_len, wordpiece_ends[b_tgt][:tgt_len], left_buffer, right_buffer
        )
        sample_cells = 0
        for u in range(tgt_len):
            v_range = ar_check.valid_time_ranges(u)
            valid_ranges[b_tgt, u, 0] = v_range[0]
            valid_ranges[b_tgt, u, 1] = v_range[1]
            total_valid += v_range[1] - v_range[0] + 1
            sample_cells += v_range[1] - v_range[0] + 1
        cells_per_sample[b_tgt] = sample_cells
    sparse_joint_enc = np.zeros((total_valid, D)).astype(dense_tensor.dtype)
    offset = 0
    for b in range(B * H):
        for u in range(U):
            st, en = valid_ranges[b_tgt][u][0], valid_ranges[b_tgt][u][1]
            sparse_joint_enc[offset : offset + (en - st) + 1, :] = dense_tensor[
                b, st : en + 1, u, :
            ]
            offset += (en - st) + 1
    return sparse_joint_enc, valid_ranges, cells_per_sample


def assert_sparse_all_close(data, gradients, ref_gradients, atol=1e-6, rtol=1e-2):
    valid_ranges = data["valid_ranges"]
    idx = 0
    for b in range(valid_ranges.size(0)):
        for u in range(valid_ranges.size(1)):
            st, en = valid_ranges[b, u, 0], valid_ranges[b, u, 1]
            np.testing.assert_allclose(
                gradients[idx : idx + (en - st + 1), :],
                ref_gradients[b, st : en + 1, u, :],
                atol=atol,
                rtol=rtol,
            )
            idx += (en - st) + 1


def compute_with_numpy_transducer(data):
    costs = NumpyTransducerLoss(
        blank=data["blank"],
    )(
        logits=data["logits"],
        logit_lengths=data["logit_lengths"],
        target_lengths=data["target_lengths"],
        targets=data["targets"],
        wordpiece_ends=data.get("wordpiece_ends", None),
        left_buffer=data.get("left_buffer", 0),
        right_buffer=data.get("right_buffer", 0),
    )

    loss = torch.sum(costs)
    loss.backward()

    costs = costs.cpu().data.numpy()
    gradients = data["logits"].saved_grad.cpu().data.numpy()

    return costs, gradients


def compute_with_pytorch_transducer(data, reuse_logits_for_grads=False):
    left_buffer = data.get("left_buffer", 0)
    right_buffer = data.get("right_buffer", 0)

    costs = RNNTLoss(
        blank=data["blank"],
        left_buffer=left_buffer,
        right_buffer=right_buffer,
        sparse=True if "logits_sparse" in data else False,
        fused_log_softmax=data.get("fused_log_softmax", True),
        reuse_logits_for_grads=reuse_logits_for_grads,
    )(
        logits=data.get("logits_sparse", data["logits"]),
        logit_lengths=data["logit_lengths"],
        target_lengths=data["target_lengths"],
        targets=data["targets"],
        wordpiece_ends=data.get("wordpiece_ends", None),
        valid_ranges=data.get("valid_ranges", None),
        cells_per_sample=data["cells_per_sample"]
        if "cells_per_sample" in data
        else None,
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu().data.numpy()
    gradients = data["logits"].saved_grad.cpu().data.numpy()
    return costs, gradients


def get_data_basic(device):
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
        dtype=torch.float,
    )
    targets = torch.tensor([[1, 2]], dtype=torch.int)
    logit_lengths = torch.tensor([2], dtype=torch.int)
    target_lengths = torch.tensor([2], dtype=torch.int)

    logits = logits.to(device=device)
    targets = targets.to(device=device)
    logit_lengths = logit_lengths.to(device=device)
    target_lengths = target_lengths.to(device=device)

    logits.requires_grad_(True)

    return logits, targets, logit_lengths, target_lengths


def get_B1_T10_U3_D4_data(
    random=False,
    left_buffer=0,
    right_buffer=0,
    sparse=False,
    dtype=np.float32,
    nan=False,
):
    B, T, U, D = 2, 10, 3, 4
    data = {}
    data["logits"] = np.random.rand(B, T, U, D).astype(dtype)
    if not random:
        data["logits"].fill(0.1)
    if nan:
        for i in range(B):
            data["logits"][i][0][0][0] = np.nan
    data["logit_lengths"] = np.array([10, 10], dtype=np.int32)
    data["target_lengths"] = np.array([2, 2], dtype=np.int32)
    data["targets"] = np.array([[1, 2], [1, 2]], dtype=np.int32)
    data["blank"] = 0
    data["wordpiece_ends"] = np.array([[0, 2, 7], [0, 2, 7]], dtype=np.int32)

    if sparse:
        sparse_joint_enc, valid_ranges, cells_per_sample = _get_sparse(
            data, data["logits"], left_buffer, right_buffer
        )
        data["logits_sparse"] = sparse_joint_enc
        data["valid_ranges"] = valid_ranges
        data["cells_per_sample"] = cells_per_sample
    return data


def get_numpy_data_B1_T2_U3_D5(dtype=np.float32):
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
    logit_lengths = np.array([2], dtype=np.int32)
    target_lengths = np.array([2], dtype=np.int32)

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
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }

    return data, ref_costs, ref_gradients


def get_numpy_data_B2_T4_U3_D3(dtype=np.float32):
    # Test from D21322854

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
    logit_lengths = np.array([4, 4], dtype=np.int32)
    target_lengths = np.array([2, 2], dtype=np.int32)

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
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }

    return data, ref_costs, ref_gradients


def get_numpy_random_data(
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

    logit_lengths = np.random.randint(low=5, high=T + 1, size=(B,), dtype=np.int32)
    target_lengths = np.random.randint(low=5, high=U + 1, size=(B,), dtype=np.int32)
    max_src_length = np.max(logit_lengths)
    max_tgt_length = np.max(target_lengths)
    targets = np.random.randint(
        low=0, high=D - 1, size=(B, max_tgt_length), dtype=np.int32
    )
    logits = np.random.random_sample(
        size=(B, max_src_length, max_tgt_length + 1, D)
    ).astype(dtype=dtype)

    return {
        "logits": logits,
        "targets": targets,
        "logit_lengths": logit_lengths,
        "target_lengths": target_lengths,
        "blank": blank,
    }


def numpy_to_torch(data, device, requires_grad=True):
    logits = torch.from_numpy(data["logits"])
    targets = torch.from_numpy(data["targets"])
    logit_lengths = torch.from_numpy(data["logit_lengths"])
    target_lengths = torch.from_numpy(data["target_lengths"])

    if "wordpiece_ends" in data:
        data["wordpiece_ends"] = torch.from_numpy(data["wordpiece_ends"]).to(device=device)
    if "nbest_wers" in data:
        data["nbest_wers"] = torch.from_numpy(data["nbest_wers"]).to(device=device)
    if "nbest_scores" in data:
        data["nbest_scores"] = torch.from_numpy(data["nbest_scores"]).to(
            device=device
        )

    logits = torch.autograd.Variable(logits, requires_grad=requires_grad)
    logit_lengths = torch.autograd.Variable(logit_lengths)
    target_lengths = torch.autograd.Variable(target_lengths)
    targets = torch.autograd.Variable(targets)

    if device == torch.device("cpu"):
        logits = logits.cpu()
    elif device == torch.device("cuda"):
        logits = logits.cuda()
    else:
        raise ValueError("unrecognized device = {}".format(device))

    def grad_hook(grad):
        logits.saved_grad = grad.clone()

    logits.register_hook(grad_hook)

    data["logits"] = logits
    data["logit_lengths"] = logit_lengths
    data["target_lengths"] = target_lengths
    data["targets"] = targets

    if "logits_sparse" in data:
        logits_sparse = torch.from_numpy(data["logits_sparse"])
        logits_sparse = torch.autograd.Variable(
            logits_sparse, requires_grad=requires_grad
        )
        logits_sparse = logits_sparse.to(device=logits.device)
        logits_sparse.register_hook(grad_hook)
        data["logits_sparse"] = logits_sparse
        valid_ranges = torch.from_numpy(data["valid_ranges"])
        valid_ranges = valid_ranges.to(device=logits.device)
        data["valid_ranges"] = valid_ranges
        cells_per_sample = torch.from_numpy(data["cells_per_sample"])
        cells_per_sample = cells_per_sample.to(device=logits.device)
        data["cells_per_sample"] = cells_per_sample
    return data


def skipIfNoTransducer(test_item):
    try:
        torch.ops.torchaudio.rnnt_loss
        return test_item
    except RuntimeError:
        return unittest.skip("torchaudio C++ extension is not compiled with RNN transducer loss")
