import warnings
from typing import List, Optional, Union

import torch

from torchaudio.functional import fftconvolve


def oscillator_bank(
    frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    sample_rate: float,
    reduction: str = "sum",
    dtype: Optional[torch.dtype] = torch.float64,
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
        dtype (torch.dtype or None, optional): The data type on which cumulative sum operation is performed.
            Default: ``torch.float64``. Pass ``None`` to disable the casting.

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

    pi2 = 2.0 * torch.pi
    freqs = frequencies * pi2 / sample_rate % pi2
    phases = torch.cumsum(freqs, dim=-2, dtype=dtype)
    if dtype is not None and freqs.dtype != dtype:
        phases = phases.to(freqs.dtype)

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


def sinc_impulse_response(cutoff: torch.Tensor, window_size: int = 513, high_pass: bool = False):
    """Create windowed-sinc impulse response for given cutoff frequencies.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        cutoff (Tensor): Cutoff frequencies for low-pass sinc filter.

        window_size (int, optional): Size of the Hamming window to apply. Must be odd.
        (Default: 513)

        high_pass (bool, optional):
            If ``True``, convert the resulting filter to high-pass.
            Otherwise low-pass filter is returned. Default: ``False``.

    Returns:
        Tensor: A series of impulse responses. Shape: `(..., window_size)`.
    """
    if window_size % 2 == 0:
        raise ValueError(f"`window_size` must be odd. Given: {window_size}")

    half = window_size // 2
    device, dtype = cutoff.device, cutoff.dtype
    idx = torch.linspace(-half, half, window_size, device=device, dtype=dtype)

    filt = torch.special.sinc(cutoff.unsqueeze(-1) * idx.unsqueeze(0))
    filt = filt * torch.hamming_window(window_size, device=device, dtype=dtype, periodic=False).unsqueeze(0)
    filt = filt / filt.sum(dim=-1, keepdim=True).abs()

    # High pass IR is obtained by subtracting low_pass IR from delta function.
    # https://courses.engr.illinois.edu/ece401/fa2020/slides/lec10.pdf
    if high_pass:
        filt = -filt
        filt[..., half] = 1.0 + filt[..., half]
    return filt


def frequency_impulse_response(magnitudes):
    """Create filter from desired frequency response

    Args:
        magnitudes: The desired frequency responses. Shape: `(..., num_fft_bins)`

    Returns:
        Tensor: Impulse response. Shape `(..., 2 * (num_fft_bins - 1))`
    """
    if magnitudes.min() < 0.0:
        # Negative magnitude does not make sense but allowing so that autograd works
        # around 0.
        # Should we raise error?
        warnings.warn("The input frequency response should not contain negative values.")
    ir = torch.fft.fftshift(torch.fft.irfft(magnitudes), dim=-1)
    device, dtype = magnitudes.device, magnitudes.dtype
    window = torch.hann_window(ir.size(-1), periodic=False, device=device, dtype=dtype).expand_as(ir)
    return ir * window


def _overlap_and_add(waveform, stride):
    num_frames, frame_size = waveform.shape[-2:]
    numel = (num_frames - 1) * stride + frame_size
    buffer = torch.zeros(waveform.shape[:-2] + (numel,), device=waveform.device, dtype=waveform.dtype)
    for i in range(num_frames):
        start = i * stride
        end = start + frame_size
        buffer[..., start:end] += waveform[..., i, :]
    return buffer


def filter_waveform(waveform: torch.Tensor, kernels: torch.Tensor, delay_compensation: int = -1):
    """Applies filters along time axis of the given waveform.

    This function applies the given filters along time axis in the following manner:

    1. Split the given waveform into chunks. The number of chunks is equal to the number of given filters.
    2. Filter each chunk with corresponding filter.
    3. Place the filtered chunks at the original indices while adding up the overlapping parts.
    4. Crop the resulting waveform so that delay introduced by the filter is removed and its length
       matches that of the input waveform.

    The following figure illustrates this.

        .. image:: https://download.pytorch.org/torchaudio/doc-assets/filter_waveform.png

    .. note::

       If the number of filters is one, then the operation becomes stationary.
       i.e. the same filtering is applied across the time axis.

    Args:
        waveform (Tensor): Shape `(..., time)`.
        kernels (Tensor): Impulse responses.
            Valid inputs are 2D tensor with shape `(num_filters, filter_length)` or
            `(N+1)`-D tensor with shape `(..., num_filters, filter_length)`, where `N` is
            the dimension of waveform.

            In case of 2D input, the same set of filters is used across channels and batches.
            Otherwise, different sets of filters are applied. In this case, the shape of
            the first `N-1` dimensions of filters must match (or be broadcastable to) that of waveform.

        delay_compensation (int): Control how the waveform is cropped after full convolution.
            If the value is zero or positive, it is interpreted as the length of crop at the
            beginning of the waveform. The value cannot be larger than the size of filter kernel.
            Otherwise the initial crop is ``filter_size // 2``.
            When cropping happens, the waveform is also cropped from the end so that the
            length of the resulting waveform matches the input waveform.

    Returns:
        Tensor: `(..., time)`.
    """
    if kernels.ndim not in [2, waveform.ndim + 1]:
        raise ValueError(
            "`kernels` must be 2 or N+1 dimension where "
            f"N is the dimension of waveform. Found: {kernels.ndim} (N={waveform.ndim})"
        )

    num_filters, filter_size = kernels.shape[-2:]
    num_frames = waveform.size(-1)

    if delay_compensation > filter_size:
        raise ValueError(
            "When `delay_compenstation` is provided, it cannot be larger than the size of filters."
            f"Found: delay_compensation={delay_compensation}, filter_size={filter_size}"
        )

    # Transform waveform's time axis into (num_filters x chunk_length) with optional padding
    chunk_length = num_frames // num_filters
    if num_frames % num_filters > 0:
        chunk_length += 1
        num_pad = chunk_length * num_filters - num_frames
        waveform = torch.nn.functional.pad(waveform, [0, num_pad], "constant", 0)
    chunked = waveform.unfold(-1, chunk_length, chunk_length)
    assert chunked.numel() >= waveform.numel()

    # Broadcast kernels
    if waveform.ndim + 1 > kernels.ndim:
        expand_shape = waveform.shape[:-1] + kernels.shape
        kernels = kernels.expand(expand_shape)

    convolved = fftconvolve(chunked, kernels)
    restored = _overlap_and_add(convolved, chunk_length)

    # Trim in a way that the number of samples are same as input,
    # and the filter delay is compensated
    if delay_compensation >= 0:
        start = delay_compensation
    else:
        start = filter_size // 2
    num_crops = restored.size(-1) - num_frames
    end = num_crops - start
    result = restored[..., start:-end]
    return result


def exp_sigmoid(
    input: torch.Tensor, exponent: float = 10.0, max_value: float = 2.0, threshold: float = 1e-7
) -> torch.Tensor:
    """Exponential Sigmoid pointwise nonlinearity.
    Implements the equation:
    ``max_value`` * sigmoid(``input``) ** (log(``exponent``)) + ``threshold``

    The output has a range of [``threshold``, ``max_value``].
    ``exponent`` controls the slope of the output.

    .. devices:: CPU CUDA

    Args:
        input (Tensor): Input Tensor
        exponent (float, optional): Exponent. Controls the slope of the output
        max_value (float, optional): Maximum value of the output
        threshold (float, optional): Minimum value of the output

    Returns:
        Tensor: Exponential Sigmoid output. Shape: same as input

    """

    return max_value * torch.pow(
        torch.nn.functional.sigmoid(input),
        torch.log(torch.tensor(exponent, device=input.device, dtype=input.dtype)),
    ) + torch.tensor(threshold, device=input.device, dtype=input.dtype)
