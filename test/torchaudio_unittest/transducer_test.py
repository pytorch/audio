import torch
from torchaudio.prototype.transducer import RNNTLoss

from torchaudio_unittest import common_utils


def get_data_basic(device):
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


def get_data_B2_T4_U3_D3(dtype=torch.float32, device="cpu"):
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
    ).reshape(2, 4, 3, 3)

    targets = torch.tensor([[1, 2], [1, 1]], dtype=torch.int32)
    src_lengths = torch.tensor([4, 4], dtype=torch.int32)
    tgt_lengths = torch.tensor([2, 2], dtype=torch.int32)

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

    logits.requires_grad_(True)
    logits = logits.to(device)

    def grad_hook(grad):
        logits.saved_grad = grad.clone()

    logits.register_hook(grad_hook)

    data = {
        "logits": logits,
        "targets": targets,
        "src_lengths": src_lengths,
        "tgt_lengths": tgt_lengths,
        "blank": blank,
    }

    return data, ref_costs, ref_gradients


def compute_with_pytorch_transducer(data):
    costs = RNNTLoss(blank=data["blank"], reduction="none")(
        acts=data["logits"],
        labels=data["targets"],
        act_lens=data["src_lengths"],
        label_lens=data["tgt_lengths"],
    )

    loss = torch.sum(costs)
    loss.backward()
    costs = costs.cpu()
    gradients = data["logits"].saved_grad.cpu()
    return costs, gradients


class TransducerTester:
    def test_basic_fp16_error(self):
        rnnt_loss = RNNTLoss()
        acts, labels, act_length, label_length = get_data_basic(self.device)
        acts = acts.to(torch.float16)
        # RuntimeError raised by log_softmax before reaching transducer's bindings
        self.assertRaises(
            RuntimeError, rnnt_loss, acts, labels, act_length, label_length
        )

    def test_basic_backward(self):
        rnnt_loss = RNNTLoss()
        acts, labels, act_length, label_length = get_data_basic(self.device)
        loss = rnnt_loss(acts, labels, act_length, label_length)
        loss.backward()

    def test_costs_and_gradients_B2_T4_U3_D3_fp32(self):

        data, ref_costs, ref_gradients = get_data_B2_T4_U3_D3(
            dtype=torch.float32, device=self.device
        )
        logits_shape = data["logits"].shape
        costs, gradients = compute_with_pytorch_transducer(data=data)

        atol, rtol = 1e-6, 1e-2
        self.assertEqual(costs, ref_costs, atol=atol, rtol=rtol)
        self.assertEqual(logits_shape, gradients.shape)
        self.assertEqual(gradients, ref_gradients, atol=atol, rtol=rtol)


@common_utils.skipIfNoExtension
class CPUTransducerTester(TransducerTester, common_utils.PytorchTestCase):
    device = "cpu"
