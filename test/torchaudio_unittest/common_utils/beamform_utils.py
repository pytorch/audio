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


def mvdr_weights_rtf_numpy(rtf, psd_n, reference_channel, diag_eps=1e-7, eps=1e-8):
    channel = rtf.shape[-1]
    eye = np.eye(channel)
    trace = np.matrix.trace(psd_n, axis1=1, axis2=2)
    epsilon = trace.real[..., None, None] * diag_eps + eps
    diag = epsilon * eye[..., :, :]
    psd_n = psd_n + diag
    numerator = np.linalg.solve(psd_n, np.expand_dims(rtf, -1)).squeeze(-1)
    denominator = np.einsum("...d,...d->...", rtf.conj(), numerator)
    beamform_weights = numerator / (np.expand_dims(denominator.real, -1) + eps)
    if isinstance(reference_channel, int):
        scale = rtf[..., reference_channel].conj()
    else:
        scale = np.einsum("...c,...c->...", rtf.conj(), reference_channel[..., None, :])
    beamform_weights = beamform_weights * scale[..., None]
    return beamform_weights


def rtf_evd_numpy(psd):
    _, v = np.linalg.eigh(psd)
    rtf = v[..., -1]
    return rtf


def rtf_power_numpy(psd_s, psd_n, reference_channel, n_iter, diagonal_loading=True, diag_eps=1e-7, eps=1e-8):
    if diagonal_loading:
        channel = psd_s.shape[-1]
        eye = np.eye(channel)
        trace = np.matrix.trace(psd_n, axis1=1, axis2=2)
        epsilon = trace.real[..., None, None] * diag_eps + eps
        diag = epsilon * eye[..., :, :]
        psd_n = psd_n + diag
    phi = np.linalg.solve(psd_n, psd_s)
    if isinstance(reference_channel, int):
        rtf = phi[..., reference_channel]
    else:
        rtf = phi @ reference_channel
    rtf = np.expand_dims(rtf, -1)
    if n_iter >= 2:
        for _ in range(n_iter - 2):
            rtf = phi @ rtf
        rtf = psd_s @ rtf
    else:
        rtf = psd_n @ rtf
    rtf = rtf.squeeze(-1)
    return rtf


def apply_beamforming_numpy(beamform_weights, specgram):
    specgram_enhanced = np.einsum("...fc,...cft->...ft", beamform_weights.conj(), specgram)
    return specgram_enhanced
