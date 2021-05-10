import torch

__all__ = [
    "RNNTLoss",
    "rnnt_loss",
]


def _rnnt_loss_alphas(
    logits,
    targets,
    logit_lengths,
    target_lengths,
    blank=-1,
    clamp=-1,
):
    """
    Compute alphas for RNN transducer loss.

    See documentation for RNNTLoss
    """
    targets = targets.to(device=logits.device)
    logit_lengths = logit_lengths.to(device=logits.device)
    target_lengths = target_lengths.to(device=logits.device)

    # make sure all int tensors are of type int32.
    targets = targets.int()
    logit_lengths = logit_lengths.int()
    target_lengths = target_lengths.int()

    return torch.ops.torchaudio.rnnt_loss_alphas(
        logits,
        targets,
        logit_lengths,
        target_lengths,
        blank,
        clamp,
    )


def _rnnt_loss_betas(
    logits,
    targets,
    logit_lengths,
    target_lengths,
    blank=-1,
    clamp=-1,
):
    """
    Compute betas for RNN transducer loss

    See documentation for RNNTLoss
    """
    targets = targets.to(device=logits.device)
    logit_lengths = logit_lengths.to(device=logits.device)
    target_lengths = target_lengths.to(device=logits.device)

    # make sure all int tensors are of type int32.
    targets = targets.int()
    logit_lengths = logit_lengths.int()
    target_lengths = target_lengths.int()

    return torch.ops.torchaudio.rnnt_loss_betas(
        logits,
        targets,
        logit_lengths,
        target_lengths,
        blank,
        clamp,
    )


class _RNNT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        targets,
        logit_lengths,
        target_lengths,
        blank=-1,
        clamp=-1,
        fused_log_softmax=True,
        reuse_logits_for_grads=True,
    ):
        """
        See documentation for RNNTLoss
        """

        # move everything to the same device.
        targets = targets.to(device=logits.device)
        logit_lengths = logit_lengths.to(device=logits.device)
        target_lengths = target_lengths.to(device=logits.device)

        # make sure all int tensors are of type int32.
        targets = targets.int()
        logit_lengths = logit_lengths.int()
        target_lengths = target_lengths.int()

        if blank < 0:  # reinterpret blank index if blank < 0.
            blank = logits.shape[-1] + blank

        costs, gradients = torch.ops.torchaudio.rnnt_loss(
            logits=logits,
            targets=targets,
            src_lengths=logit_lengths,
            tgt_lengths=target_lengths,
            blank=blank,
            clamp=clamp,
            fused_log_smax=fused_log_softmax,
            reuse_logits_for_grads=reuse_logits_for_grads,
        )

        ctx.grads = gradients

        return costs

    @staticmethod
    def backward(ctx, output_gradients):
        output_gradients = output_gradients.view(-1, 1, 1, 1).to(ctx.grads)
        ctx.grads.mul_(output_gradients).to(ctx.grads)

        return (
            ctx.grads,  # logits
            None,  # targets
            None,  # logit_lengths
            None,  # target_lengths
            None,  # blank
            None,  # clamp
            None,  # fused_log_softmax
            None,  # reuse_logits_for_grads
        )


def rnnt_loss(
    logits,
    targets,
    logit_lengths,
    target_lengths,
    blank=-1,
    clamp=-1,
    fused_log_softmax=True,
    reuse_logits_for_grads=True,
):
    """
    Compute the RNN Transducer Loss.

    The RNN Transducer loss (`Graves 2012 <https://arxiv.org/pdf/1211.3711.pdf>`__) extends the CTC loss by defining
    a distribution over output sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        logits (Tensor): Tensor of dimension (batch, time, target, class) containing output from joiner
        targets (Tensor): Tensor of dimension (batch, max target length) containing targets with zero padded
        logit_lengths (Tensor): Tensor of dimension (batch) containing lengths of each sequence from encoder
        target_lengths (Tensor): Tensor of dimension (batch) containing lengths of targets for each sequence
        blank (int, opt): blank label (Default: ``-1``)
        clamp (float): clamp for gradients (Default: ``-1``)
        runtime_check (bool): whether to do sanity check during runtime. (Default: ``False``)
        fused_log_softmax (bool): set to False if calling log_softmax outside loss (Default: ``True``)
        reuse_logits_for_grads (bool): whether to save memory by reusing logits memory for grads (Default: ``True``)
    """
    if not fused_log_softmax:
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        reuse_logits_for_grads = (
            False  # softmax needs the original logits value
        )

    cost = _RNNT.apply(
        logits,
        targets,
        logit_lengths,
        target_lengths,
        blank,
        clamp,
        fused_log_softmax,
        reuse_logits_for_grads,
    )
    return cost


class RNNTLoss(torch.nn.Module):
    """
    Compute the RNN Transducer Loss.

    The RNN Transducer loss (`Graves 2012 <https://arxiv.org/pdf/1211.3711.pdf>`__) extends the CTC loss by defining
    a distribution over output sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        blank (int, opt): blank label (Default: ``-1``)
        clamp (float): clamp for gradients (Default: ``-1``)
        fused_log_softmax (bool): set to False if calling log_softmax outside loss (Default: ``True``)
        reuse_logits_for_grads (bool): whether to save memory by reusing logits memory for grads (Default: ``True``)
    """

    def __init__(
        self,
        blank=-1,
        clamp=-1,
        fused_log_softmax=True,
        reuse_logits_for_grads=True,
    ):
        super().__init__()
        self.blank = blank
        self.clamp = clamp
        self.fused_log_softmax = fused_log_softmax
        self.reuse_logits_for_grads = reuse_logits_for_grads

    def forward(
        self,
        logits,
        targets,
        logit_lengths,
        target_lengths,
    ):
        """
        Args:
            logits (Tensor): Tensor of dimension (batch, time, target, class) containing output from joiner
            targets (Tensor): Tensor of dimension (batch, max target length) containing targets with zero padded
            logit_lengths (Tensor): Tensor of dimension (batch) containing lengths of each sequence from encoder
            target_lengths (Tensor): Tensor of dimension (batch) containing lengths of targets for each sequence
        """
        return rnnt_loss(
            logits,
            targets,
            logit_lengths,
            target_lengths,
            self.blank,
            self.clamp,
            self.fused_log_softmax,
            self.reuse_logits_for_grads,
        )
