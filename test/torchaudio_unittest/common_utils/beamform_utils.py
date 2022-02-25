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


def mvdr_weights_souden_numpy(psd_s, psd_n, reference_channel, diag_eps=1e-7, eps=1e-8):
    channel = psd_s.shape[-1]
    eye = np.eye(channel)
    trace = np.matrix.trace(psd_n, axis1=1, axis2=2)
    epsilon = trace.real[..., None, None] * diag_eps + eps
    diag = epsilon * eye[..., :, :]
    psd_n = psd_n + diag
    numerator = np.linalg.solve(psd_n, psd_s)  # psd_n.inv() @ psd_s
    numerator_trace = np.matrix.trace(numerator, axis1=1, axis2=2)
    ws = numerator / (numerator_trace[..., None, None] + eps)
    if isinstance(reference_channel, int):
        beamform_weights = ws[..., :, reference_channel]
    else:
        beamform_weights = np.einsum("...c,...c->...", ws, reference_channel[..., None, None, :])
    return beamform_weights
