import numpy as np


def psd_numpy(specgram, mask=None, normalize=True, eps=1e-10):
    specgram_transposed = np.swapaxes(specgram, 0, 1)
    psd = np.einsum("...ct,...et->...tce", specgram_transposed, specgram_transposed.conj())
    if mask is not None:
        if normalize:
            mask_normmalized = mask / (mask.sum(axis=-1, keepdims=True) + eps)
        else:
            mask_normmalized = mask
        psd = psd * mask_normmalized[..., None, None]
    psd = psd.sum(axis=-3)
    return psd
