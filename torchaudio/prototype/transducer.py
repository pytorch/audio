import torch


def compute_rnnt_alphas(
    logits,
    targets,
    src_lengths,
    tgt_lengths,
    blank=-1,
    clamp=-1,
):
    """Wrapper function to compute alphas for RNNT loss.

    This can also be used as a backpointer table, to backtrack {from (T-1, U-1) to (0,0)} to find the best cost path.
    Also, enables easy unit-testing of alphas with numpy

    Args:
        logits (Tensor): Tensor of (B, max_T, max_U, D) containing output from joiner
        targets (Tensor): Tensor of (B, max_U - 1) containing targets with zero padded
        src_lengths (Tensor): Tensor of (B) containing lengths of each sequence from encoder
        tgt_lengths (Tensor): Tensor of (B) containing lengths of targets for each sequence
        blank (int): blank label. (Default: ``-1``)
        clamp (float): clamp for gradients (Default: ``-1``)
    """
    targets = targets.to(device=logits.device)
    src_lengths = src_lengths.to(device=logits.device)
    tgt_lengths = tgt_lengths.to(device=logits.device)

    # make sure all int tensors are of type int32.
    targets = targets.int()
    src_lengths = src_lengths.int()
    tgt_lengths = tgt_lengths.int()

    return torch.ops.torchaudio.compute_transducer_alphas(
        logits,
        targets,
        src_lengths,
        tgt_lengths,
        blank,
        clamp,
    )


def compute_rnnt_betas(
    logits,
    targets,
    src_lengths,
    tgt_lengths,
    blank=-1,
    clamp=-1,
):
    """Wrapper function to compute betas for RNNT loss.

    Enables easy unit-testing of alphas with numpy

    Args:
        logits (Tensor): Tensor of (B, max_T, max_U, D) containing output from joiner
        targets (Tensor): Tensor of (B, max_U - 1) containing targets with zero padded
        src_lengths (Tensor): Tensor of (B) containing lengths of each sequence from encoder
        tgt_lengths (Tensor): Tensor of (B) containing lengths of targets for each sequence
        blank (int): blank label. (Default: ``-1``)
        clamp (float): clamp for gradients (Default: ``-1``)
    """
    targets = targets.to(device=logits.device)
    src_lengths = src_lengths.to(device=logits.device)
    tgt_lengths = tgt_lengths.to(device=logits.device)

    # make sure all int tensors are of type int32.
    targets = targets.int()
    src_lengths = src_lengths.int()
    tgt_lengths = tgt_lengths.int()

    return torch.ops.torchaudio.compute_transducer_betas(
        logits,
        targets,
        src_lengths,
        tgt_lengths,
        blank,
        clamp,
    )


class _RNNT(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        targets,
        src_lengths,
        tgt_lengths,
        blank=-1,
        clamp=-1,
        runtime_check=False,
        fused_log_smax=True,
        reuse_logits_for_grads=True,
    ):
        """
        See documentation for RNNTLoss
        """

        # move everything to the same device.
        targets = targets.to(device=logits.device)
        src_lengths = src_lengths.to(device=logits.device)
        tgt_lengths = tgt_lengths.to(device=logits.device)

        # make sure all int tensors are of type int32.
        targets = targets.int()
        src_lengths = src_lengths.int()
        tgt_lengths = tgt_lengths.int()

        if blank < 0:  # reinterpret blank index if blank < 0.
            blank = logits.shape[-1] + blank

        if runtime_check:
            check_inputs(
                logits=logits,
                targets=targets,
                src_lengths=src_lengths,
                tgt_lengths=tgt_lengths,
                blank=blank,
            )

        costs, gradients = torch.ops.torchaudio.compute_transducer_loss(
            logits=logits,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank=blank,
            clamp=clamp,
            fused_log_smax=fused_log_smax,
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
            None,  # src_lengths
            None,  # tgt_lengths
            None,  # blank
            None,  # clamp
            None,  # runtime_check
            None,  # fused_log_smax
            None,  # reuse_logits_for_grads
        )


def rnnt_loss(
    logits,
    targets,
    src_lengths,
    tgt_lengths,
    blank=-1,
    clamp=-1,
    runtime_check=False,
    fused_log_smax=True,
    reuse_logits_for_grads=True
):
    """Compute the RNN Transducer Loss.

    The RNN Transducer loss (`Graves 2012 <https://arxiv.org/pdf/1211.3711.pdf>`__) extends the CTC loss by defining
    a distribution over output sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        logits (Tensor): Tensor of (B, max_T, max_U, D) containing output from joiner
        targets (Tensor): Tensor of (B, max_U - 1) containing targets with zero padded
        src_lengths (Tensor): Tensor of (B) containing lengths of each sequence from encoder
        tgt_lengths (Tensor): Tensor of (B) containing lengths of targets for each sequence
        blank (int): blank label. (Default: ``-1``)
        clamp (float): clamp for gradients (Default: ``-1``)
        runtime_check (bool): whether to do sanity check during runtime. (Default: ``False``)
        fused_log_smax (bool): set to False if calling log_softmax outside loss (Default: ``True``)
        reuse_logits_for_grads (bool): whether to save memory by reusing logits memory for grads (Default: ``True``)
    """

    if not fused_log_smax:
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        reuse_logits_for_grads = (
            False  # softmax needs the original logits value
        )

    cost = _RNNT.apply(
        logits,
        targets,
        src_lengths,
        tgt_lengths,
        blank,
        clamp,
        runtime_check,
        fused_log_smax,
        reuse_logits_for_grads,
    )
    return cost


class RNNTLoss(torch.nn.Module):
    """Compute the RNN Transducer Loss.

    The RNN Transducer loss (`Graves 2012 <https://arxiv.org/pdf/1211.3711.pdf>`__) extends the CTC loss by defining
    a distribution over output sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        blank (int): blank label. (Default: ``-1``)
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
        fused_log_smax=True,
        reuse_logits_for_grads=True,
    ):
        super().__init__()
        self.blank = blank
        self.clamp = clamp
        self.runtime_check = runtime_check
        self.fused_log_smax = fused_log_smax
        self.reuse_logits_for_grads = reuse_logits_for_grads

    def forward(
        self,
        logits,
        targets,
        src_lengths,
        tgt_lengths,
    ):
        """
        Args:
            logits (Tensor): Tensor of (B, max_T, max_U, D) containing output from joiner
            targets (Tensor): Tensor of (B, max_U - 1) containing targets with zero padded
            src_lengths (Tensor): Tensor of (B) containing lengths of each sequence from encoder
            tgt_lengths (Tensor): Tensor of (B) containing lengths of targets for each sequence
        """

        # Do not use fused log softmax if explicitly specified using
        # fused_log_smax=False (for example to do Min WER training)
        # For below cases, we call log_softmax outside of loss
        if not self.fused_log_smax:
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            self.reuse_logits_for_grads = (
                False  # softmax needs the original logits value
            )

        cost = _RNNT.apply(
            logits,
            targets,
            src_lengths,
            tgt_lengths,
            self.blank,
            self.clamp,
            self.runtime_check,
            self.fused_log_smax,
            self.reuse_logits_for_grads,
        )
        return cost


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


def check_inputs(logits, targets, src_lengths, tgt_lengths, blank):
    check_device(logits, "logits", targets, "targets")
    check_device(logits, "logits", targets, "src_lengths")
    check_device(logits, "logits", targets, "tgt_lengths")

    check_type(logits, torch.float32, "logits")
    check_type(targets, torch.int32, "targets")
    check_type(src_lengths, torch.int32, "src_lengths")
    check_type(tgt_lengths, torch.int32, "tgt_lengths")

    check_contiguous(logits, "logits")
    check_contiguous(targets, "targets")
    check_contiguous(tgt_lengths, "tgt_lengths")
    check_contiguous(src_lengths, "src_lengths")

    check_dim(logits, 4, "logits")
    check_dim(targets, 2, "targets")
    check_dim(src_lengths, 1, "src_lengths")
    check_dim(tgt_lengths, 1, "tgt_lengths")

    check_equal(
        src_lengths.shape[0], "src_lengths.shape[0]", logits.shape[0], "logits.shape[0]"
    )
    check_equal(
        tgt_lengths.shape[0], "tgt_lengths.shape[0]", logits.shape[0], "logits.shape[0]"
    )
    check_equal(
        targets.shape[0], "targets.shape[0]", logits.shape[0], "logits.shape[0]"
    )
    check_equal(
        targets.shape[1],
        "targets.shape[1]",
        torch.max(tgt_lengths),
        "torch.max(tgt_lengths)",
    )
    check_equal(
        logits.shape[1],
        "logits.shape[1]",
        torch.max(src_lengths),
        "torch.max(src_lengths)",
    )
    check_equal(
        logits.shape[2],
        "logits.shape[2]",
        torch.max(tgt_lengths) + 1,
        "torch.max(tgt_lengths) + 1",
    )

    if blank < 0 or blank >= logits.shape[-1]:
        raise ValueError(
            "blank ({}) must be within [0, logits.shape[-1]={})".format(
                blank, logits.shape[-1]
            )
        )
