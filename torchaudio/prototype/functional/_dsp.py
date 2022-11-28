import warnings
from typing import List, Optional, Union

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
        Using ``torch.float64`` can work around this.

        The following figure shows the difference between ``torch.float32`` and
        ``torch.float64`` when generating a sin wave of constant frequency and amplitude
        with sample rate 8000 [Hz].
        Notice that ``torch.float32`` version shows artifacts that are not seen in
        ``torch.float64`` version.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/oscillator_precision.png

    Args:
        frequencies (Tensor): Sample-wise oscillator frequencies (Hz). Shape `(..., time, N)`.
        amplitudes (Tensor): Sample-wise oscillator amplitude. Shape: `(..., time, N)`.
        sample_rate (float): Sample rate
        reduction (str): Reduction to perform.
            Valid values are ``"sum"``, ``"mean"`` or ``"none"``. Default: ``"sum"``

    Returns:
        Tensor:
            The resulting waveform.

            If ``reduction`` is ``"none"``, then the shape is
            `(..., time, N)`, otherwise the shape is `(..., time)`.
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
    phases = torch.cumsum(freqs, dim=-2)

    waveform = amplitudes * torch.sin(phases)
    if reduction == "sum":
        return waveform.sum(-1)
    if reduction == "mean":
        return waveform.mean(-1)
    return waveform


def adsr_envelope(
    num_frames: int,
    *,
    attack: float = 0.0,
    hold: float = 0.0,
    decay: float = 0.0,
    sustain: float = 1.0,
    release: float = 0.0,
    n_decay: int = 2,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    """Generate ADSR Envelope

    .. devices:: CPU CUDA

    Args:
        num_frames (int): The number of output frames.
        attack (float, optional):
            The relative *time* it takes to reach the maximum level from
            the start. (Default: ``0.0``)
        hold (float, optional):
            The relative *time* the maximum level is held before
            it starts to decay. (Default: ``0.0``)
        decay (float, optional):
            The relative *time* it takes to sustain from
            the maximum level. (Default: ``0.0``)
        sustain (float, optional): The relative *level* at which
            the sound should sustain. (Default: ``1.0``)

            .. Note::
               The duration of sustain is derived as `1.0 - (The sum of attack, hold, decay and release)`.

        release (float, optional): The relative *time* it takes for the sound level to
            reach zero after the sustain. (Default: ``0.0``)
        n_decay (int, optional): The degree of polynomial decay. Default: ``2``.
        dtype (torch.dtype, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default
            (see :py:func:`torch.set_default_tensor_type`).
        device (torch.device, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :py:func:`torch.set_default_tensor_type`).
            device will be the CPU for CPU tensor types and the current CUDA
            device for CUDA tensor types.

    Returns:
        Tensor: ADSR Envelope. Shape: `(num_frames, )`

    Example
        .. image:: https://download.pytorch.org/torchaudio/doc-assets/adsr_examples.png

    """
    if not 0 <= attack <= 1:
        raise ValueError(f"The value of `attack` must be within [0, 1]. Found: {attack}")
    if not 0 <= decay <= 1:
        raise ValueError(f"The value of `decay` must be within [0, 1]. Found: {decay}")
    if not 0 <= sustain <= 1:
        raise ValueError(f"The value of `sustain` must be within [0, 1]. Found: {sustain}")
    if not 0 <= hold <= 1:
        raise ValueError(f"The value of `hold` must be within [0, 1]. Found: {hold}")
    if not 0 <= release <= 1:
        raise ValueError(f"The value of `release` must be within [0, 1]. Found: {release}")
    if attack + decay + release + hold > 1:
        raise ValueError("The sum of `attack`, `hold`, `decay` and `release` must not exceed 1.")

    nframes = num_frames - 1
    num_a = int(nframes * attack)
    num_h = int(nframes * hold)
    num_d = int(nframes * decay)
    num_r = int(nframes * release)

    # Initialize with sustain
    out = torch.full((num_frames,), float(sustain), device=device, dtype=dtype)

    # attack
    if num_a > 0:
        torch.linspace(0.0, 1.0, num_a + 1, out=out[: num_a + 1])

    # hold
    if num_h > 0:
        out[num_a : num_a + num_h + 1] = 1.0

    # decay
    if num_d > 0:
        # Compute: sustain + (1.0 - sustain) * (linspace[1, 0] ** n_decay)
        i = num_a + num_h
        decay = out[i : i + num_d + 1]
        torch.linspace(1.0, 0.0, num_d + 1, out=decay)
        decay **= n_decay
        decay *= 1.0 - sustain
        decay += sustain

    # sustain is handled by initialization

    # release
    if num_r > 0:
        torch.linspace(sustain, 0, num_r + 1, out=out[-num_r - 1 :])

    return out


def extend_pitch(
    base: torch.Tensor,
    pattern: Union[int, List[float], torch.Tensor],
):
    """Extend the given time series values with multipliers of them.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given a series of fundamental frequencies (pitch), this function appends
    its harmonic overtones or inharmonic partials.

    Args:
        base (torch.Tensor):
            Base time series, like fundamental frequencies (Hz). Shape: `(..., time, 1)`.
        pattern (int, list of floats or torch.Tensor):
            If ``int``, the number of pitch series after the operation.
            `pattern - 1` tones are added, so that the resulting Tensor contains
            up to `pattern`-th overtones of the given series.

            If list of float or ``torch.Tensor``, it must be one dimensional,
            representing the custom multiplier of the fundamental frequency.

    Returns:
        Tensor: Oscillator frequencies (Hz). Shape: `(..., time, num_tones)`.

    Example
        >>> # fundamental frequency
        >>> f0 = torch.linspace(1, 5, 5).unsqueeze(-1)
        >>> f0
        tensor([[1.],
                [2.],
                [3.],
                [4.],
                [5.]])
        >>> # Add harmonic overtones, up to 3rd.
        >>> f = extend_pitch(f0, 3)
        >>> f.shape
        torch.Size([5, 3])
        >>> f
        tensor([[ 1.,  2.,  3.],
                [ 2.,  4.,  6.],
                [ 3.,  6.,  9.],
                [ 4.,  8., 12.],
                [ 5., 10., 15.]])
        >>> # Add custom (inharmonic) partials.
        >>> f = extend_pitch(f0, torch.tensor([1, 2.1, 3.3, 4.5]))
        >>> f.shape
        torch.Size([5, 4])
        >>> f
        tensor([[ 1.0000,  2.1000,  3.3000,  4.5000],
                [ 2.0000,  4.2000,  6.6000,  9.0000],
                [ 3.0000,  6.3000,  9.9000, 13.5000],
                [ 4.0000,  8.4000, 13.2000, 18.0000],
                [ 5.0000, 10.5000, 16.5000, 22.5000]])
    """
    if isinstance(pattern, torch.Tensor):
        mult = pattern
    elif isinstance(pattern, int):
        mult = torch.linspace(1.0, float(pattern), pattern, device=base.device, dtype=base.dtype)
    else:
        mult = torch.tensor(pattern, dtype=base.dtype, device=base.device)
    h_freq = base @ mult.unsqueeze(0)
    return h_freq
