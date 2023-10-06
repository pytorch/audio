import math
import warnings
from typing import Optional

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


def _hz_to_octs(freqs, tuning=0.0, bins_per_octave=12):
    a440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    return torch.log2(freqs / (a440 / 16))


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


def chroma_filterbank(
    sample_rate: int,
    n_freqs: int,
    n_chroma: int,
    *,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: Optional[float] = 2.0,
    norm: int = 2,
    base_c: bool = True,
):
    """Create a frequency-to-chroma conversion matrix. Implementation adapted from librosa.

    Args:
        sample_rate (int): Sample rate.
        n_freqs (int): Number of input frequencies.
        n_chroma (int): Number of output chroma.
        tuning (float, optional): Tuning deviation from A440 in fractions of a chroma bin. (Default: 0.0)
        ctroct (float, optional): Center of Gaussian dominance window to weight filters by, in octaves. (Default: 5.0)
        octwidth (float or None, optional): Width of Gaussian dominance window to weight filters by, in octaves.
            If ``None``, then disable weighting altogether. (Default: 2.0)
        norm (int, optional): order of norm to normalize filter bank by. (Default: 2)
        base_c (bool, optional): If True, then start filter bank at C. Otherwise, start at A. (Default: True)

    Returns:
        torch.Tensor: Chroma filter bank, with shape `(n_freqs, n_chroma)`.
    """
    # Skip redundant upper half of frequency range.
    freqs = torch.linspace(0, sample_rate // 2, n_freqs)[1:]
    freq_bins = n_chroma * _hz_to_octs(freqs, bins_per_octave=n_chroma, tuning=tuning)
    freq_bins = torch.cat((torch.tensor([freq_bins[0] - 1.5 * n_chroma]), freq_bins))
    freq_bin_widths = torch.cat(
        (
            torch.maximum(freq_bins[1:] - freq_bins[:-1], torch.tensor(1.0)),
            torch.tensor([1]),
        )
    )

    # (n_freqs, n_chroma)
    D = freq_bins.unsqueeze(1) - torch.arange(0, n_chroma)

    n_chroma2 = round(n_chroma / 2)

    # Project to range [-n_chroma/2, n_chroma/2 - 1]
    D = torch.remainder(D + n_chroma2, n_chroma) - n_chroma2

    fb = torch.exp(-0.5 * (2 * D / torch.tile(freq_bin_widths.unsqueeze(1), (1, n_chroma))) ** 2)
    fb = torch.nn.functional.normalize(fb, p=norm, dim=1)

    if octwidth is not None:
        fb *= torch.tile(
            torch.exp(-0.5 * (((freq_bins.unsqueeze(1) / n_chroma - ctroct) / octwidth) ** 2)),
            (1, n_chroma),
        )

    if base_c:
        fb = torch.roll(fb, -3 * (n_chroma // 12), dims=1)

    return fb
