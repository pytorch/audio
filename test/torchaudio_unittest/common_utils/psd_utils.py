from typing import Optional

import numpy as np
import torch


def psd_numpy(
    X: np.array, mask: Optional[np.array], multi_mask: bool = False, normalize: bool = True, eps: float = 1e-15
) -> np.array:
    X_conj = np.conj(X)
    psd_X = np.einsum("...cft,...eft->...ftce", X, X_conj)
    if mask is not None:
        if multi_mask:
            mask = mask.mean(axis=-3)
        if normalize:
            mask = mask / (mask.sum(axis=-1, keepdims=True) + eps)
        psd = psd_X * mask[..., None, None]
    else:
        psd = psd_X

    psd = psd.sum(axis=-3)

    return torch.tensor(psd, dtype=torch.cdouble)
