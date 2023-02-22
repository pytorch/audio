from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_contrastive_loss(x, mask_indices, targets, neg_is_pos, reduce, logit_temp = 0.1):
    x = (
        x[mask_indices]
        .view(x.size(0), -1, x.size(-1))
        .unsqueeze(0)
        .expand(targets.shape)
    )
    logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).float()
    logits /= logit_temp
    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")
    target = logits.new_zeros(logits.size(1) * logits.size(2), dtype=torch.long)
    logits = logits.transpose(0, 2)
    logits = logits.reshape(-1, logits.size(-1))
    loss = F.cross_entropy(
        logits,
        target,
        reduction="sum" if reduce else "none",
    )
    sample_size = target.numel()
    return loss, sample_size, logits

def wav2vec2_loss(
    x: Tensor,
    mask_indices: Tensor,
    y: Tensor,
    negatives: Tensor, 
    reduce: Optional[bool]=True
) -> Tuple[Tensor, float]:
    """Compute Wav2Vec2 loss.

    Args:
        x (Tensor): The masked sequences of probability distribution.
        mask_indices (Tensor): The mask indices.
        y (Tensor): The ys, prior to negative sampling.
        negatives (Tensor): The negative samples.
        reduce (bool, optional): Use "sum" as reduction for cross-entropy loss (Default: ``True``).

    Returns:
        (Tensor, float)
        Tensor: The desired loss Tensor.
        float: Sample size according to mask_indices
    """
    assert y is not None
    assert mask_indices is not None
    assert mask_indices.sum() == y.shape[0] * y.shape[1]

    # 4. compute targets
    neg_is_pos = (y == negatives).all(-1)
    y = y.unsqueeze(0)
    targets = torch.cat([y, negatives], dim=0)

    # 5. compute losses
    loss, sample_size, _ = compute_contrastive_loss(
        x, mask_indices, targets, neg_is_pos, reduce
    )
    loss = loss.float()

    return loss, sample_size
