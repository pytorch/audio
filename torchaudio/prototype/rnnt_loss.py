import torch
from torch import Tensor

__all__ = [
    "RNNTLoss",
    "rnnt_loss",
]


def rnnt_loss(
    logits: Tensor,
    targets: Tensor,
    logit_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = -1,
    clamp: float = -1,
    fused_log_softmax: bool = True,
    reuse_logits_for_grads: bool = True,
):
    """Compute the RNN Transducer loss from *Sequence Transduction with Recurrent Neural Networks*
    [:footcite:`graves2012sequence`].

    The RNN Transducer loss extends the CTC loss by defining a distribution over output
    sequences of all lengths, and by jointly modelling both input-output and output-output
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

    Returns:
        Tensor: Loss with the reduction option applied. If ``reduction`` is  ``'none'``, then size (batch),
            otherwise scalar.
    """
    if not fused_log_softmax:
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        reuse_logits_for_grads = (
            False  # softmax needs the original logits value
        )

    if blank < 0:  # reinterpret blank index if blank < 0.
        blank = logits.shape[-1] + blank

    costs, gradients = torch.ops.torchaudio.rnnt_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=blank,
        clamp=clamp,
        fused_log_softmax=fused_log_softmax,
        reuse_logits_for_grads=reuse_logits_for_grads,)

    return costs


class RNNTLoss(torch.nn.Module):
    """Compute the RNN Transducer loss from *Sequence Transduction with Recurrent Neural Networks*
    [:footcite:`graves2012sequence`].

    The RNN Transducer loss extends the CTC loss by defining a distribution over output
    sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        blank (int, opt): blank label (Default: ``-1``)
        clamp (float): clamp for gradients (Default: ``-1``)
        fused_log_softmax (bool): set to False if calling log_softmax outside loss (Default: ``True``)
        reuse_logits_for_grads (bool): whether to save memory by reusing logits memory for grads (Default: ``True``)
    """

    def __init__(
        self,
        blank: int = -1,
        clamp: float = -1.,
        fused_log_softmax: bool = True,
        reuse_logits_for_grads: bool = True,
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

        Returns:
            Tensor: Loss with the reduction option applied. If ``reduction`` is  ``'none'``, then size (batch),
                otherwise scalar.
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
