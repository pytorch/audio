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
    reduction: str = "mean",
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
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. (Default: ``'mean'``)

    Returns:
        Tensor: Loss with the reduction option applied. If ``reduction`` is  ``'none'``, then size (batch),
            otherwise scalar.
    """
    if reduction not in ['none', 'mean', 'sum']:
        raise ValueError("reduction should be one of 'none', 'mean', or 'sum'")

    if blank < 0:  # reinterpret blank index if blank < 0.
        blank = logits.shape[-1] + blank

    costs, _ = torch.ops.torchaudio.rnnt_loss(
        logits=logits,
        targets=targets,
        logit_lengths=logit_lengths,
        target_lengths=target_lengths,
        blank=blank,
        clamp=clamp,
    )

    if reduction == 'mean':
        return costs.mean()
    elif reduction == 'sum':
        return costs.sum()

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
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. (Default: ``'mean'``)
    """

    def __init__(
        self,
        blank: int = -1,
        clamp: float = -1.,
        reduction: str = "mean",
    ):
        super().__init__()
        self.blank = blank
        self.clamp = clamp
        self.reduction = reduction

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
            self.reduction
        )
