from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_contrastive_loss(
    x: Tensor,
    mask_indices: Tensor,
    targets: Tensor,
    neg_is_pos: Tensor,
    reduction: str = "none",
    logit_temp: float = 0.1,
):
    """
    Computes the contrastive loss used in Wav2Vec2 loss function.

    Args:
        x (Tensor): Input embeddings of shape `(batch_size, sequence_length, hidden_size)`.
        mask_indices (Tensor): Indices to mask negative samples of shape `(batch_size, sequence_length)`.
        targets (Tensor): Labels indicating positive samples.
            Tensor of shape `(num_negative + 1, batch, sequence_length, hidden_size)`.
        neg_is_pos (Tensor): Boolean tensor indicating whether negative samples should be treated as positives.
            Tensor of shape `(batch, sequence_length)`.
        reduction (str): Reduction type ("sum" or "none").
        logit_temp (float, optional): Temperature scaling factor for logits, defaults to 0.1.

    Returns:
        The computed contrastive loss and sample size
    """

    x = x[mask_indices].view(x.size(0), -1, x.size(-1)).unsqueeze(0).expand(targets.shape)
    logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).float()
    logits /= logit_temp
    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")
    target = logits.new_zeros(logits.size(1) * logits.size(2), dtype=torch.long, device=logits.device)
    logits = logits.transpose(0, 2)
    logits = logits.reshape(-1, logits.size(-1))
    loss = F.cross_entropy(
        logits,
        target,
        reduction=reduction,
    )
    sample_size = target.numel()
    return loss, sample_size


def wav2vec2_loss(
    x: Tensor, mask_indices: Tensor, positives: Tensor, negatives: Tensor, reduction: str = "none"
) -> Tuple[Tensor, float]:
    """Compute Wav2Vec2 loss.

    Args:
        x (Tensor): The masked sequences of Wav2Vec 2.0 model.
            Tensor of shape `(batch_size, sequence_length, hidden_size)`.
        mask_indices (Tensor): The mask indices. Tensor of shape `(batch_size, sequence_length)`
        positives (Tensor): The positives, prior to negative sampling.
            Tensor of shape `(batch_size, masked_sequence_length, hidden_size)`
        negatives (Tensor): The negative samples.
            Tensor of shape `(num_negative, batch_size, masked_sequence_length, hidden_size)`
        reduction (str): Use "sum" as reduction for cross-entropy loss (Default: ``none``)

    Returns:
        (Tensor, float)
        Tensor: The desired loss Tensor.
        float: Sample size according to mask_indices
    """
    assert positives is not None
    assert mask_indices is not None
    assert mask_indices.sum() == positives.shape[0] * positives.shape[1]

    neg_is_pos = (positives == negatives).all(-1)
    positives = positives.unsqueeze(0)
    targets = torch.cat([positives, negatives], dim=0)

    loss, sample_size = compute_contrastive_loss(x, mask_indices, targets, neg_is_pos, reduction)

    return loss, sample_size
