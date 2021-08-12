from typing import Callable, Tuple
import torch
from torch import Tensor
from torch.autograd import gradcheck
from torchaudio_unittest.common_utils import (
    TestBaseMixin,
)
from torchaudio.prototype.rnnt_loss import RNNTLoss, rnnt_loss
from parameterized import parameterized
from .utils import (
    get_B1_T10_U3_D4_data,
    get_B2_T4_U3_D3_data,
    get_B1_T2_U3_D5_data
)
from .numpy_transducer import NumpyTransducerLoss


class Autograd(TestBaseMixin):
    @staticmethod
    def get_data(data_func, device):
        data = data_func()
        if type(data) == tuple:
            data = data[0]
        return data

    def assert_grad(
            self,
            loss: Callable[..., Tensor],
            inputs: Tuple[torch.Tensor],
            *,
            enable_all_grad: bool = True,
    ):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(dtype=self.dtype, device=self.device)
                if enable_all_grad:
                    i.requires_grad = True
            inputs_.append(i)
        # gradcheck with float32 requires higher atol and epsilon
        assert gradcheck(loss, inputs, eps=1e-3, atol=1e-3, nondet_tol=0.)

    @parameterized.expand([
        (get_B1_T10_U3_D4_data, ),
        (get_B2_T4_U3_D3_data, ),
        (get_B1_T2_U3_D5_data, ),
    ])
    def test_RNNTLoss_gradcheck(self, data_func):
        data = self.get_data(data_func, self.device)
        inputs = (
            data["logits"].to(self.dtype),
            data["targets"],
            data["logit_lengths"],
            data["target_lengths"],
        )
        loss = RNNTLoss(blank=data["blank"])

        self.assert_grad(loss, inputs, enable_all_grad=False)

    @parameterized.expand([
        (get_B1_T10_U3_D4_data, ),
        (get_B2_T4_U3_D3_data, ),
        (get_B1_T2_U3_D5_data, ),
    ])
    def test_rnnt_loss_gradcheck(self, data_func):
        data = self.get_data(data_func, self.device)
        inputs = (
            data["logits"].to(self.dtype),  # logits
            data["targets"],                # targets
            data["logit_lengths"],          # logit_lengths
            data["target_lengths"],         # target_lengths
            data["blank"],                  # blank
            -1,                             # clamp
        )

        self.assert_grad(rnnt_loss, inputs, enable_all_grad=False)

    @parameterized.expand([
        (get_B1_T10_U3_D4_data, ),
        (get_B2_T4_U3_D3_data, ),
        (get_B1_T2_U3_D5_data, ),
    ])
    def test_np_transducer_gradcheck(self, data_func):
        data = self.get_data(data_func, self.device)
        inputs = (
            data["logits"].to(self.dtype),
            data["logit_lengths"],
            data["target_lengths"],
            data["targets"],
        )
        loss = NumpyTransducerLoss(blank=data["blank"])

        self.assert_grad(loss, inputs, enable_all_grad=False)
