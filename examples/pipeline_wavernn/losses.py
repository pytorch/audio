import torch
from torch.nn import functional as F


def log_sum_exp(x):
    r""" Numerically stable log_sum_exp implementation that prevents overflow
    """

    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def MoLLoss(y_hat, y, num_classes=65536, log_scale_min=None, reduce=True):
    r""" Discretized mixture of logistic distributions loss

    Adapted from wavenet vocoder
    (https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py)
    Explanation of loss (https://github.com/Rayhane-mamah/Tacotron-2/issues/155)

    Args:
        y_hat (Tensor): Predicted output (n_batch x n_time x n_channel)
        y (Tensor): Target (n_batch x n_time x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each minibatch.

    Returns
        Tensor: loss
    """

    if log_scale_min is None:
        log_scale_min = torch.log(torch.as_tensor(1e-14)).item()

    assert y_hat.dim() == 3
    assert y_hat.size(-1) % 3 == 0

    nr_mix = y_hat.size(-1) // 3

    # unpack parameters (n_batch, n_time, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix: 2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix: 3 * nr_mix], min=log_scale_min)

    # (n_batch x n_time x 1) to (n_batch x n_time x num_mixtures)
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1.0 / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(F.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - F.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * torch.log(
        torch.clamp(cdf_delta, min=1e-12)
    ) + (1.0 - inner_inner_cond) * (
        log_pdf_mid - torch.log(torch.as_tensor((num_classes - 1) / 2)).item()
    )
    inner_cond = (y > 0.999).float()
    inner_out = (
        inner_cond * log_one_minus_cdf_min + (1.0 - inner_cond) * inner_inner_out
    )
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.0 - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)
