import torch
from torch.autograd import Function
from torch.nn import Module
from torchaudio._internal import (
    module_utils as _mod_utils,
)

__all__ = [
    "rnnt_loss",
    "RNNTLoss",
]


class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank, reduction):
        """
        See documentation for RNNTLoss.
        """

        device = acts.device

        acts = acts.to("cpu")
        labels = labels.to("cpu")
        act_lens = act_lens.to("cpu")
        label_lens = label_lens.to("cpu")

        loss_func = torch.ops.torchaudio.rnnt_loss

        grads = torch.zeros_like(acts)
        minibatch_size = acts.size(0)
        costs = torch.zeros(minibatch_size, dtype=acts.dtype)

        loss_func(acts, labels, act_lens, label_lens, costs, grads, blank, 0)

        if reduction in ["sum", "mean"]:
            costs = costs.sum().unsqueeze_(-1)
            if reduction == "mean":
                costs /= minibatch_size
                grads /= minibatch_size

        costs = costs.to(device)
        ctx.grads = grads.to(device)

        return costs

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul_(grad_output), None, None, None, None, None


@_mod_utils.requires_module("torchaudio._torchaudio")
def rnnt_loss(acts, labels, act_lens, label_lens, blank=0, reduction="mean"):
    """Compute the RNN Transducer Loss.

    The RNN Transducer loss (`Graves 2012 <https://arxiv.org/pdf/1211.3711.pdf>`__) extends the CTC loss by defining
    a distribution over output sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    The implementation uses `warp-transducer <https://github.com/HawkAaron/warp-transducer>`__.

    Args:
        acts (Tensor): Tensor of dimension (batch, time, label, class) containing output from network
            before applying ``torch.nn.functional.log_softmax``.
        labels (Tensor): Tensor of dimension (batch, max label length) containing the labels padded by zero
        act_lens (Tensor): Tensor of dimension (batch) containing the length of each output sequence
        label_lens (Tensor): Tensor of dimension (batch) containing the length of each output sequence
        blank (int): blank label. (Default: ``0``)
        reduction (string): If ``'sum'``, the output losses will be summed.
            If ``'mean'``, the output losses will be divided by the target lengths and
            then the mean over the batch is taken. If ``'none'``, no reduction will be applied.
            (Default: ``'mean'``)
    """

    # NOTE manually done log_softmax for CPU version,
    # log_softmax is computed within GPU version.
    acts = torch.nn.functional.log_softmax(acts, -1)
    return _RNNT.apply(acts, labels, act_lens, label_lens, blank, reduction)


@_mod_utils.requires_module("torchaudio._torchaudio")
class RNNTLoss(Module):
    """Compute the RNN Transducer Loss.

    The RNN Transducer loss (`Graves 2012 <https://arxiv.org/pdf/1211.3711.pdf>`__) extends the CTC loss by defining
    a distribution over output sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    The implementation uses `warp-transducer <https://github.com/HawkAaron/warp-transducer>`__.

    Args:
        blank (int): blank label. (Default: ``0``)
        reduction (string): If ``'sum'``, the output losses will be summed.
            If ``'mean'``, the output losses will be divided by the target lengths and
            then the mean over the batch is taken. If ``'none'``, no reduction will be applied.
            (Default: ``'mean'``)
    """

    def __init__(self, blank=0, reduction="mean"):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.loss = _RNNT.apply

    def forward(self, acts, labels, act_lens, label_lens):
        """
        Args:
            acts (Tensor): Tensor of dimension (batch, time, label, class) containing output from network
                before applying ``torch.nn.functional.log_softmax``.
            labels (Tensor): Tensor of dimension (batch, max label length) containing the labels padded by zero
            act_lens (Tensor): Tensor of dimension (batch) containing the length of each output sequence
            label_lens (Tensor): Tensor of dimension (batch) containing the length of each output sequence
        """

        # NOTE manually done log_softmax for CPU version,
        # log_softmax is computed within GPU version.
        acts = torch.nn.functional.log_softmax(acts, -1)
        return self.loss(acts, labels, act_lens, label_lens, self.blank, self.reduction)
