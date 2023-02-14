from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def hubert_loss(
    logit_m: Optional[Tensor],
    logit_u: Optional[Tensor],
    feature_penalty: Tensor,
    label: Optional[Tensor] = None,
    masked_weight: float = 1.0,
    unmasked_weight: float = 0.0,
    feature_weight: float = 10.0,
    reduction: str = "sum",
) -> Tuple[Tensor, float]:
    """Compute the cross-entropy loss on HuBERT masked and non-masked logits.

    Args:
        logit_m (Tensor or None): The masked logit Tensor of dimension `(masked_frames, final_dim)`.
        logit_u (Tensor or None): The non-masked logit Tensor of dimension `(unmasked_frames, final_dim)`.
        feature_penalty (Tensor): The feature mean value for additional penalty loss.
        masked_weight (float, optional): The weight for masked cross-entropy loss (Default: ``1.0``).
        unmasked_weight (float, optional): The weight for non-masked cross-entropy loss (Default: ``0.0``).
        feature_weight (float, optional): The weight for feature penalty loss (Default: ``10.0``).
        reduction (str, optional): The reduction method for cross-entropy loss (Default: ``"sum"``).

    Returns:
        (Tensor, float)
        Tensor: The desired loss Tensor.
        float: Number of frames used in loss computation.
    """
    num_frame = 0.0
    loss = 0.0
    if logit_m is not None:
        target_m = torch.zeros(logit_m.shape[0], dtype=torch.long, device=logit_m.device)
        loss_m = F.cross_entropy(logit_m, target_m, reduction=reduction)
        loss += loss_m * masked_weight
        num_frame += logit_m.shape[0]
    if logit_u is not None:
        target_u = torch.zeros(logit_u.shape[0], dtype=torch.long, device=logit_m.device)
        loss_u = F.cross_entropy(logit_u, target_u, reduction=reduction)
        loss += loss_u * unmasked_weight
        num_frame += logit_u.shape[0]
    loss += feature_penalty * feature_weight * num_frame
    return loss, num_frame
