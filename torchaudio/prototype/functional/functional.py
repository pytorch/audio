import math
import warnings
from typing import Tuple

import torch
from torchaudio.functional.functional import _create_triangular_filterbank


def _hz_to_bark(freqs: float, bark_scale: str = "traunmuller") -> float:
    r"""Convert Hz to Barks.

    Args:
        freqs (float): Frequencies in Hz
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        barks (float): Frequency in Barks
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError('bark_scale should be one of "schroeder", "traunmuller" or "wang".')

    if bark_scale == "wang":
        return 6.0 * math.asinh(freqs / 600.0)
    elif bark_scale == "schroeder":
        return 7.0 * math.asinh(freqs / 650.0)
    # Traunmuller Bark scale
    barks = ((26.81 * freqs) / (1960.0 + freqs)) - 0.53
    # Bark value correction
    if barks < 2:
        barks += 0.15 * (2 - barks)
    elif barks > 20.1:
        barks += 0.22 * (barks - 20.1)

    return barks


def _bark_to_hz(barks: torch.Tensor, bark_scale: str = "traunmuller") -> torch.Tensor:
    """Convert bark bin numbers to frequencies.

    Args:
        barks (torch.Tensor): Bark frequencies
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        freqs (torch.Tensor): Barks converted in Hz
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError('bark_scale should be one of "traunmuller", "schroeder" or "wang".')

    if bark_scale == "wang":
        return 600.0 * torch.sinh(barks / 6.0)
    elif bark_scale == "schroeder":
        return 650.0 * torch.sinh(barks / 7.0)
    # Bark value correction
    if any(barks < 2):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif any(barks > 20.1):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22

    # Traunmuller Bark scale
    freqs = 1960 * ((barks + 0.53) / (26.28 - barks))

    return freqs


def barkscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_barks: int,
    sample_rate: int,
    bark_scale: str = "traunmuller",
) -> torch.Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    .. image:: https://download.pytorch.org/torchaudio/doc-assets/bark_fbanks.png
        :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_barks (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        torch.Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_barks``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * barkscale_fbanks(A.size(-1), ...)``.

    """

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate bark freq bins
    m_min = _hz_to_bark(f_min, bark_scale=bark_scale)
    m_max = _hz_to_bark(f_max, bark_scale=bark_scale)

    m_pts = torch.linspace(m_min, m_max, n_barks + 2)
    f_pts = _bark_to_hz(m_pts, bark_scale=bark_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one bark filterbank has all zero values. "
            f"The value for `n_barks` ({n_barks}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb


def forced_align(
    emission: torch.Tensor, label: torch.Tensor, blank_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forced alignment given the CTC emission and ground truth transcript.

    Args:
        emission (torch.Tensor): CTC emission output. Tensor with dimensions `(time, vocabulary)`.
        label (torch.Tensor): Transcript label. Tensor with dimension `(token,)`.
        blank_id (int): The index of blank symbol in CTC emission.

    Returns:
        Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
            torch.Tensor: Path recording the label in each frame of emission.
            torch.Tensor: Token indices.
            torch.Tensor: Time indices.
            torch.Tensor: Scores of each frame in emission.

    Note:
        The sequence length of `emission` must satisfy:

        .. math::
            L_{\\text{emission}} \\ge L_{\\text{label}} + N_{\\text{repeat}}

        where :math:`N_{\\text{repeat}}` is the number of consecutively repeated tokens.
        For example, in str `"aabbc"`, the number of repeats are `2`.
    """
    T = emission.size(0)  # num frames
    L = len(label)  # label length
    R = sum([1 if label[i] == label[i - 1] else 0 for i in range(1, L)])  # number of repeats
    S = 2 * L + 1  # We construct a trellis with blanks inserted between the labels.

    assert T >= L + R, f"label length too long for CTC T:{T}, L:{L}, R:{R}"

    # not all labels can be explored for a given time step.
    # start and end labels track it.
    start = 0 if (T - (L + R)) > 0 else 1
    end = 1 if S == 1 else 2

    # we need to only maintain two frames for alpha
    # alpha for current time frame depends only on previous time frame
    # alpha stands for the cumulative proabilities of all paths reaching a
    # particular label and time step
    alphas = [-float("inf") for _ in range(2 * S)]

    # backptr stores the index offset of the best path at current position
    # At each time step, we explore if the best path came from i, i-1 or i-2 label
    # of previous time step and store backptr as 0, 1 or 2 respectively.
    backptr = [-10000] * (T * S)
    for i in range(start, end):
        # Note that we use blanks in between the label - "<b> c <b> a <b> t <b>"
        # So, all even indices are blank and odd indices are actual target labels
        label_idx = blank_id if (i % 2 == 0) else label[i // 2]
        # performance: is it better to convert emission to list ?
        # or convert alphas to tensor ?
        alphas[i] = emission[0, label_idx]

    # Iterate through each time frame
    for t in range(1, T):

        # Calculate the smallest and largest possible index of the target that this
        # time could be
        if T - t <= L + R:
            if (start % 2 == 1) and (label[start // 2] != label[start // 2 + 1]):
                start = start + 1
            start = start + 1
        if t <= L + R:
            if (end % 2 == 0) and (end < 2 * L) and (label[end // 2 - 1] != label[end // 2]):
                end = end + 1
            end = end + 1

        #         print(T, t, L, R, start, end)

        startloop = start
        idx1 = (t % 2) * S
        idx2 = ((t - 1) % 2) * S

        if start == 0:
            alphas[idx1] = alphas[idx2] + emission[t, blank_id]
            backptr[t * S] = 0
            startloop += 1

        for i in range(startloop, end):
            x0 = alphas[i + idx2]
            x1 = alphas[(i - 1) + idx2]
            x2 = -float("inf")

            label_idx = blank_id if (i % 2 == 0) else label[i // 2]

            # In CTC, the optimal path may optionally chose to skip a blank label.
            # x2 represents skipping a letter, and can only happen if we're not
            # currently on a blank_label, and we're not on a repeat letter
            # (i != 1) just ensures we don't access labels[i - 2] if its i < 2
            if i % 2 != 0 and i != 1 and label[i // 2] != label[i // 2 - 1]:
                x2 = alphas[(i - 2) + idx2]

            result = 0.0
            if x2 > x1 and x2 > x0:
                result = x2
                backptr[i + t * S] = 2
            elif x1 > x0 and x1 > x2:
                result = x1
                backptr[i + t * S] = 1
            else:
                result = x0
                backptr[i + t * S] = 0

            alphas[i + idx1] = result + emission[t, label_idx]

    idx1 = ((T - 1) % 2) * S
    ltr_idx = S - 1 if alphas[idx1 + S - 1] > alphas[idx1 + S - 2] else S - 2

    # path stores the token index for each time step after force alignment.
    paths = [-1] * T
    token_indices = []
    time_indices = []
    scores = []
    for t in range(T - 1, -1, -1):
        lbl_idx = blank_id if (ltr_idx % 2 == 0) else label[ltr_idx // 2]
        paths[t] = lbl_idx
        if lbl_idx != blank_id:
            token_indices.append(ltr_idx // 2)
            time_indices.append(t)
            scores.append(emission[t, lbl_idx].exp().item())
        ltr_idx -= backptr[(t * S) + ltr_idx]

    return paths, torch.tensor(token_indices), torch.tensor(time_indices), torch.tensor(scores)
