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
        runtime_check=False,
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

        if runtime_check:
            check_inputs(
                logits=logits,
                targets=targets,
                logit_lengths=logit_lengths,
                target_lengths=target_lengths,
                blank=blank,
            )

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
            None,  # runtime_check
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
    runtime_check=False,
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
        fused_log_smax (bool): set to False if calling log_softmax outside loss (Default: ``True``)
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
        runtime_check,
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
        runtime_check (bool): whether to do sanity check during runtime. (Default: ``False``)
        fused_log_smax (bool): set to False if calling log_softmax outside loss (Default: ``True``)
        reuse_logits_for_grads (bool): whether to save memory by reusing logits memory for grads (Default: ``True``)
    """

    def __init__(
        self,
        blank=-1,
        clamp=-1,
        runtime_check=False,
        fused_log_softmax=True,
        reuse_logits_for_grads=True,
    ):
        super().__init__()
        self.blank = blank
        self.clamp = clamp
        self.runtime_check = runtime_check
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
            self.runtime_check,
            self.fused_log_softmax,
            self.reuse_logits_for_grads,
        )


def check_type(var, t, name):
    if var.dtype is not t:
        raise TypeError("{} must be {}".format(name, t))


def check_contiguous(var, name):
    if not var.is_contiguous():
        raise ValueError("{} must be contiguous".format(name))


def check_dim(var, dim, name):
    if len(var.shape) != dim:
        raise ValueError("{} must be {}D".format(name, dim))


def check_equal(var1, name1, var2, name2):
    if var1 != var2:
        raise ValueError(
            "`{}` ({}) must equal to ".format(name1, var1)
            + "`{}` ({})".format(name2, var2)
        )


def check_device(var1, name1, var2, name2):
    if var1.device != var2.device:
        raise ValueError(
            "`{}` ({}) must be on the same ".format(name1, var1.device.type)
            + "device as `{}` ({})".format(name2, var2.device.type)
        )


def check_inputs(logits, targets, logit_lengths, target_lengths, blank):
    check_device(logits, "logits", targets, "targets")
    check_device(logits, "logits", targets, "logit_lengths")
    check_device(logits, "logits", targets, "target_lengths")

    check_type(logits, torch.float32, "logits")
    check_type(targets, torch.int32, "targets")
    check_type(logit_lengths, torch.int32, "logit_lengths")
    check_type(target_lengths, torch.int32, "target_lengths")

    check_contiguous(logits, "logits")
    check_contiguous(targets, "targets")
    check_contiguous(target_lengths, "target_lengths")
    check_contiguous(logit_lengths, "logit_lengths")

    check_dim(logits, 4, "logits")
    check_dim(targets, 2, "targets")
    check_dim(logit_lengths, 1, "logit_lengths")
    check_dim(target_lengths, 1, "target_lengths")

    check_equal(
        logit_lengths.shape[0], "logit_lengths.shape[0]", logits.shape[0], "logits.shape[0]"
    )
    check_equal(
        target_lengths.shape[0], "target_lengths.shape[0]", logits.shape[0], "logits.shape[0]"
    )
    check_equal(
        targets.shape[0], "targets.shape[0]", logits.shape[0], "logits.shape[0]"
    )
    check_equal(
        targets.shape[1],
        "targets.shape[1]",
        torch.max(target_lengths),
        "torch.max(target_lengths)",
    )
    check_equal(
        logits.shape[1],
        "logits.shape[1]",
        torch.max(logit_lengths),
        "torch.max(logit_lengths)",
    )
    check_equal(
        logits.shape[2],
        "logits.shape[2]",
        torch.max(target_lengths) + 1,
        "torch.max(target_lengths) + 1",
    )

    if blank < 0 or blank >= logits.shape[-1]:
        raise ValueError(
            "blank ({}) must be within [0, logits.shape[-1]={})".format(
                blank, logits.shape[-1]
            )
        )
