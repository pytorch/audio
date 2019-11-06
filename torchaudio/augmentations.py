import math
import torch

from . import functional as F

__all__ = [
    'TimeStretch',
    'FrequencyMasking',
    'TimeMasking'
]


class TimeStretch(torch.jit.ScriptModule):
    r"""Stretch stft in time without modifying pitch for a given rate.

    Args:
        hop_length (int): Number audio of frames between STFT columns. (Default: ``n_fft // 2``)
        n_freq (int, optional): number of filter banks from stft. (Default: ``201``)
        fixed_rate (float): rate to speed up or slow down by.
            If None is provided, rate must be passed to the forward method. (Default: ``None``)
    """
    __constants__ = ['fixed_rate']

    def __init__(self, hop_length=None, n_freq=201, fixed_rate=None):
        super(TimeStretch, self).__init__()

        n_fft = (n_freq - 1) * 2
        hop_length = hop_length if hop_length is not None else n_fft // 2
        self.fixed_rate = fixed_rate
        phase_advance = torch.linspace(0, math.pi * hop_length, n_freq)[..., None]

        self.phase_advance = torch.jit.Attribute(phase_advance, torch.Tensor)

    @torch.jit.script_method
    def forward(self, complex_specgrams, overriding_rate=None):
        # type: (Tensor, Optional[float]) -> Tensor
        r"""
        Args:
            complex_specgrams (Tensor): complex spectrogram (*, channel, freq, time, complex=2)
            overriding_rate (float or None): speed up to apply to this batch.
                If no rate is passed, use ``self.fixed_rate``

        Returns:
            (Tensor): Stretched complex spectrogram of dimension (*, channel, freq, ceil(time/rate), complex=2)
        """
        assert complex_specgrams.size(-1) == 2, "complex_specgrams should be a complex tensor, shape (*, complex=2)"

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
        r"""
        Args:
            specgram (torch.Tensor): Tensor of dimension (*, channel, freq, time)

        Returns:
            torch.Tensor: Masked spectrogram of dimensions (*, channel, freq, time)
        """

        # if iid_masks flag marked and specgram has a batch dimension
        if self.iid_masks and specgram.dim() == 4:
            return F.mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + 1)
        else:
            shape = specgram.size()
            specgram = specgram.reshape([-1] + list(shape[-2:]))
            specgram = F.mask_along_axis(specgram, self.mask_param, mask_value, self.axis)

            return specgram.reshape(shape[:-2] + specgram.shape[-2:])


class FrequencyMasking(_AxisMasking):
    r"""
    Apply masking to a spectrogram in the frequency domain.
    Args:
        freq_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, freq_mask_param).
        iid_masks (bool): weather to apply the same mask to all
            the examples/channels in the batch. (Default: False)
    """

    def __init__(self, freq_mask_param, iid_masks=False):
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)


class TimeMasking(_AxisMasking):
    r"""
    Apply masking to a spectrogram in the time domain.
    Args:
        time_mask_param (int): maximum possible length of the mask.
            Indices uniformly sampled from [0, time_mask_param).
        iid_masks (bool): weather to apply the same mask to all
            the examples/channels in the batch. Defaults to False.
    """

    def __init__(self, time_mask_param, iid_masks=False):
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)
