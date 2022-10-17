import warnings

import torch


def oscillator_bank(
    frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    sample_rate: float,
    reduction: str = "sum",
) -> torch.Tensor:
    """Synthesize waveform from the given instantaneous frequencies and amplitudes.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Note:
        The phase information of the output waveform is found by taking the cumulative sum
        of the given instantaneous frequencies (``frequencies``).
        This incurs roundoff error when the data type does not have enough precision.
        Using ``torch.float64`` can workaround this.

        The following figure shows the difference between ``torch.float32`` and
        ``torch.float64`` when generating a sin wave of constant frequency and amplitude
        with sample rate 8000 [Hz].
        Notice that ``torch.float32`` version shows artifacts that are not seen in
        ``torch.float64`` version.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/oscillator_precision.png

    Args:
        frequencies (Tensor): Sample-wise oscillator frequencies (Hz). Shape `[..., time, N]`.
        amplitudes (Tensor): Sample-wise oscillator amplitude. Shape: `[..., time, N]`.
        sample_rate (float): Sample rate
        reduction (str): Reduction to perform.
            Valid values are ``"sum"``, ``"mean"`` or ``"none"``. Default: ``"sum"``

    Returns:
        Tensor:
            The resulting waveform.

            If ``reduction`` is ``"none"``, then the shape is
            `[..., time, N]`, otherwise the shape is `[..., time]`.
    """
    if frequencies.shape != amplitudes.shape:
        raise ValueError(
            "The shapes of `frequencies` and `amplitudes` must match. "
            f"Found: {frequencies.shape} and {amplitudes.shape} respectively."
        )
    reductions = ["sum", "mean", "none"]
    if reduction not in reductions:
        raise ValueError(f"The value of reduction must be either {reductions}. Found: {reduction}")

    invalid = torch.abs(frequencies) >= sample_rate / 2
    if torch.any(invalid):
        warnings.warn(
            "Some frequencies are above nyquist frequency. "
            "Setting the corresponding amplitude to zero. "
            "This might cause numerically unstable gradient."
        )
        amplitudes = torch.where(invalid, 0.0, amplitudes)

    # Note:
    # In magenta/ddsp, there is an option to reduce the number of summation to reduce
    # the accumulation error.
    # https://github.com/magenta/ddsp/blob/7cb3c37f96a3e5b4a2b7e94fdcc801bfd556021b/ddsp/core.py#L950-L955
    # It mentions some performance penalty.
    # In torchaudio, a simple way to work around is to use float64.
    # We might add angular_cumsum if it turned out to be undesirable.
    pi2 = 2.0 * torch.pi
    freqs = frequencies * pi2 / sample_rate % pi2
    phases = torch.cumsum(freqs, axis=-2)

    waveform = amplitudes * torch.sin(phases)
    if reduction == "sum":
        return waveform.sum(-1)
    if reduction == "mean":
        return waveform.mean(-1)
    return waveform
