import math
import torch

from . import functional as F


class TimeStretch(torch.jit.ScriptModule):
    r"""Stretch stft in time without modifying pitch for a given rate.

    Args:
        hop_length (int): Number audio of frames between STFT columns.
        num_freqs (int, optional): number of filter banks from stft.
        fixed_rate (float): rate to speed up or slow down by.
            Defaults to None (in which case a rate must be
            passed to the forward method per batch).
    """
    __constants__ = ['fixed_rate']

    def __init__(self, hop_length=200, num_freqs=201, fixed_rate=None):
        super(TimeStretch, self).__init__()

        self.fixed_rate = fixed_rate
        phase_advance = torch.linspace(0, math.pi * hop_length, num_freqs)[..., None]

        self.phase_advance = torch.jit.Attribute(phase_advance, torch.Tensor)

    @torch.jit.script_method
    def forward(self, complex_specgrams, overriding_rate=None):
        # type: (Tensor, Optional[float]) -> Tensor
        r"""
        Args:
            complex_specgrams (Tensor): complex spectrogram
                (*, channel, freq, time, complex=2)
            overriding_rate (float or None): speed up to apply to this batch.
                If no rate is passed, use self.fixed_rate
        Returns:
            (Tensor): (*, channel, num_freqs, ceil(time/rate), complex=2)
        """
        if overriding_rate is None:
            rate = self.fixed_rate
            if rate is None:
                raise ValueError("If no fixed_rate is specified"
                                 ", must pass a valid rate to the forward method.")
        else:
            rate = overriding_rate

        if rate == 1.0:
            return complex_specgrams

        shape = complex_specgrams.size()
        complex_specgrams = complex_specgrams.reshape([-1] + list(shape[-3:]))
        complex_specgrams = F.phase_vocoder(complex_specgrams, rate, self.phase_advance)

        return complex_specgrams.reshape(shape[:-3] + complex_specgrams.shape[-3:])


@torch.jit.script
def mask_along_axis_iid(specgram, mask_param, mask_value, axis):
    # type: (Tensor, int, float, int) -> Tensor
    r"""
    Apply a mask along ``axis``. Mask will be applied from ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectogram (batch, channel, num_freqs, time)
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (2 -> frequency, 3 -> time)
    """

    if axis != 2 and axis != 3:
        raise ValueError('Only Frequency and Time masking are supported')

    value = torch.rand(specgram.shape[:2]) * mask_param
    min_value = torch.rand(specgram.shape[:2]) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).unsqueeze(-1).float()
    mask_end = (min_value.long() + value.long()).unsqueeze(-1).float()

    mask = torch.arange(0, specgram.size(axis)).repeat(specgram.size(0), specgram.size(1), 1).float()

    specgram = specgram.transpose(2, axis)
    specgram[(mask >= mask_start) & (mask < mask_end)] = torch.tensor(mask_value)
    specgram = specgram.transpose(2, axis)

    return specgram


@torch.jit.script
def mask_along_axis(specgram, mask_param, mask_value, axis):
    # type: (Tensor, int, float, int) -> Tensor
    r"""
    Apply a mask along ``axis``. Mask will be applied from ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectogram (batch, channel, num_freqs, time)
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)
    """

    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()

    if axis == 1:
        specgram[:, mask_start:mask_end] = mask_value
    elif axis == 2:
        specgram[:, :, mask_start:mask_end] = mask_value
    else:
        raise ValueError('Only Frequency and Time masking is supported')

    return specgram


class _AxisMasking(torch.jit.ScriptModule):
    r"""
    Apply masking to a spectrogram.
    Args:
        mask_param (int): Maximum possible length of the mask
        axis: What dimension the mask is applied on
        iid_masks (bool): Applies iid masks to each of the examples in the batch dimension
    """
    __constants__ = ['mask_param', 'axis', 'iid_masks']

    def __init__(self, mask_param, axis, iid_masks):

        super(_AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks

    @torch.jit.script_method
    def forward(self, specgram, mask_value=0.):
        # type: (Tensor, float) -> Tensor
        # if iid_masks flag marked and specgram has a batch dimension
        if self.iid_masks and specgram.dim() == 4:
            return mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + 1)
        else:
            shape = specgram.size()
            specgram = specgram.reshape([-1] + list(shape[-2:]))
            specgram = mask_along_axis(specgram, self.mask_param, mask_value, self.axis)

            return specgram.reshape(shape[:-2] + specgram.shape[-2:])


class FrequencyMasking(_AxisMasking):
    r"""
    Apply masking to a spectrogram in the frequency domain.
    Args:
        freq_mask_param (int): maximum possible length of the mask.
            Uniformly sampled from [0, freq_mask_param).
        iid_masks (bool): weather to apply the same mask to all
            the examples/channels in the batch. Defaults to False.
    """

    def __init__(self, freq_mask_param, iid_masks=False):
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)


class TimeMasking(_AxisMasking):
    """
    Apply masking to a spectrogram in the time domain.
    Args:
        time_mask_param (int): maximum possible length of the mask.
            Uniformly sampled from [0, time_mask_param).
        iid_masks (bool): weather to apply the same mask to all
            the examples/channels in the batch. Defaults to False.
    """

    def __init__(self, time_mask_param, iid_masks=False):
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)
