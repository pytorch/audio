from __future__ import absolute_import, division, print_function, unicode_literals
import math
import fractions
import random
import torch
import torchaudio

__all__ = [
    'get_mel_banks',
    'inverse_mel_scale',
    'inverse_mel_scale_scalar',
    'mel_scale',
    'mel_scale_scalar',
    'spectrogram',
    'fbank',
    'mfcc',
    'vtln_warp_freq',
    'vtln_warp_mel_freq',
    'resample_waveform',
]

# numeric_limits<float>::epsilon() 1.1920928955078125e-07
EPSILON = torch.tensor(torch.finfo(torch.float).eps, dtype=torch.get_default_dtype())
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001

# window types
HAMMING = 'hamming'
HANNING = 'hanning'
POVEY = 'povey'
RECTANGULAR = 'rectangular'
BLACKMAN = 'blackman'
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]


def _next_power_of_2(x):
    r"""Returns the smallest power of 2 that is greater than x
    """
    return 1 if x == 0 else 2**(x - 1).bit_length()


def _get_strided(waveform, window_size, window_shift, snip_edges):
    r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
    representing how the window is shifted along the waveform. Each row is a frame.

    Args:
        waveform (torch.Tensor): Tensor of size ``num_samples``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.

    Returns:
        torch.Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 1
    num_samples = waveform.size(0)
    strides = (window_shift * waveform.stride(0), waveform.stride(0))

    if snip_edges:
        if num_samples < window_size:
            return torch.empty((0, 0))
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            # torch.nn.functional.pad returns [2,1,0,1,2] for 'reflect'
            # but we want [2, 1, 0, 0, 1, 2]
            pad_left = reversed_waveform[-pad:]
            waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
        else:
            # pad is negative so we want to trim the waveform at the front
            waveform = torch.cat((waveform[-pad:], pad_right), dim=0)

    sizes = (m, window_size)
    return waveform.as_strided(sizes, strides)


def _feature_window_function(window_type, window_size, blackman_coeff):
    r"""Returns a window function with the given type and size
    """
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46)
    elif window_type == POVEY:
        # like hanning but goes to zero at edges
        return torch.hann_window(window_size, periodic=False).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, dtype=torch.get_default_dtype())
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
        # can't use torch.blackman_window as they use different coefficients
        return blackman_coeff - 0.5 * torch.cos(a * window_function) + \
            (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
    else:
        raise Exception('Invalid window type ' + window_type)


def _get_log_energy(strided_input, epsilon, energy_floor):
    r"""Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
    if energy_floor == 0.0:
        return log_energy
    else:
        return torch.max(log_energy,
                         torch.tensor(math.log(energy_floor), dtype=torch.get_default_dtype()))


def _get_waveform_and_window_properties(waveform, channel, sample_frequency, frame_shift,
                                        frame_length, round_to_power_of_two, preemphasis_coefficient):
    r"""Gets the waveform and window properties
    """
    channel = max(channel, 0)
    assert channel < waveform.size(0), ('Invalid channel %d for size %d' % (channel, waveform.size(0)))
    waveform = waveform[channel, :]  # size (n)
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), ('choose a window size %d that is [2, %d]' % (window_size, len(waveform)))
    assert 0 < window_shift, '`window_shift` must be greater than 0'
    assert padded_window_size % 2 == 0, 'the padded ' \
        '`window_size` must be divisible by two. use `round_to_power_of_two` or change `frame_length`'
    assert 0. <= preemphasis_coefficient <= 1.0, '`preemphasis_coefficient` must be between [0,1]'
    assert sample_frequency > 0, '`sample_frequency` must be greater than zero'
    return waveform, window_shift, window_size, padded_window_size


def _get_window(waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
                snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient):
    r"""Gets a window and its log energy

    Returns:
        strided_input (torch.Tensor): size (m, ``padded_window_size``)
        signal_log_energy (torch.Tensor): size (m)
    """
    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        # Returns a random number strictly between 0 and 1
        x = torch.max(EPSILON, torch.rand(strided_input.shape))
        rand_gauss = torch.sqrt(-2 * x.log()) * torch.cos(2 * math.pi * x)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        # Compute the log energy of each row/frame before applying preemphasis and
        # window function
        signal_log_energy = _get_log_energy(strided_input, EPSILON, energy_floor)  # size (m)

    if preemphasis_coefficient != 0.0:
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (1, 0), mode='replicate').squeeze(0)  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    # Apply window_function to each row/frame
    window_function = _feature_window_function(
        window_type, window_size, blackman_coeff).unsqueeze(0)  # size (1, window_size)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (0, padding_right), mode='constant', value=0).squeeze(0)

    # Compute energy after window function (not the raw one)
    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, EPSILON, energy_floor)  # size (m)

    return strided_input, signal_log_energy


def _subtract_column_mean(tensor, subtract_mean):
    # subtracts the column mean of the tensor size (m, n) if subtract_mean=True
    # it returns size (m, n)
    if subtract_mean:
        col_means = torch.mean(tensor, dim=0).unsqueeze(0)
        tensor = tensor - col_means
    return tensor


def spectrogram(
        waveform, blackman_coeff=0.42, channel=-1, dither=1.0, energy_floor=0.0,
        frame_length=25.0, frame_shift=10.0, min_duration=0.0,
        preemphasis_coefficient=0.97, raw_energy=True, remove_dc_offset=True,
        round_to_power_of_two=True, sample_frequency=16000.0, snip_edges=True,
        subtract_mean=False, window_type=POVEY):
    r"""Create a spectrogram from a raw audio signal. This matches the input/output of Kaldi's
    compute-spectrogram-feats.

    Args:
        waveform (torch.Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        channel (int): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``1.0``)
        energy_floor (float): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``0.0``)
        frame_length (float): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float): Frame shift in milliseconds (Default: ``10.0``)
        min_duration (float): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        preemphasis_coefficient (float): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset: Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        window_type (str): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman') (Default: ``'povey'``)

    Returns:
        torch.Tensor: A spectrogram identical to what Kaldi would output. The shape is
        (m, ``padded_window_size // 2 + 1``) where m is calculated in _get_strided
    """
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0)

    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
        snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient)

    # size (m, padded_window_size // 2 + 1, 2)
    fft = torch.rfft(strided_input, 1, normalized=False, onesided=True)

    # Convert the FFT into a power spectrum
    power_spectrum = torch.max(fft.pow(2).sum(2), EPSILON).log()  # size (m, padded_window_size // 2 + 1)
    power_spectrum[:, 0] = signal_log_energy

    power_spectrum = _subtract_column_mean(power_spectrum, subtract_mean)
    return power_spectrum


def inverse_mel_scale_scalar(mel_freq):
    # type: (float) -> float
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def inverse_mel_scale(mel_freq):
    return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)


def mel_scale_scalar(freq):
    # type: (float) -> float
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq):
    return 1127.0 * (1.0 + freq / 700.0).log()


def vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq,
                   vtln_warp_factor, freq):
    r"""This computes a VTLN warping function that is not the same as HTK's one,
    but has similar inputs (this function has the advantage of never producing
    empty bins).

    This function computes a warp function F(freq), defined between low_freq
    and high_freq inclusive, with the following properties:
        F(low_freq) == low_freq
        F(high_freq) == high_freq
    The function is continuous and piecewise linear with two inflection
        points.
    The lower inflection point (measured in terms of the unwarped
        frequency) is at frequency l, determined as described below.
    The higher inflection point is at a frequency h, determined as
        described below.
    If l <= f <= h, then F(f) = f/vtln_warp_factor.
    If the higher inflection point (measured in terms of the unwarped
        frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
        Since (by the last point) F(h) == h/vtln_warp_factor, then
        max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
        h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
          = vtln_high_cutoff * min(1, vtln_warp_factor).
    If the lower inflection point (measured in terms of the unwarped
        frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
        This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
                            = vtln_low_cutoff * max(1, vtln_warp_factor)
    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        freq (torch.Tensor): given frequency in Hz

    Returns:
        torch.Tensor: Freq after vtln warp
    """
    assert vtln_low_cutoff > low_freq, 'be sure to set the vtln_low option higher than low_freq'
    assert vtln_high_cutoff < high_freq, 'be sure to set the vtln_high option lower than high_freq [or negative]'
    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l  # F(l)
    Fh = scale * h  # F(h)
    assert l > low_freq and h < high_freq
    # slope of left part of the 3-piece linear function
    scale_left = (Fl - low_freq) / (l - low_freq)
    # [slope of center part is just "scale"]

    # slope of right part of the 3-piece linear function
    scale_right = (high_freq - Fh) / (high_freq - h)

    res = torch.empty_like(freq)

    outside_low_high_freq = torch.lt(freq, low_freq) | torch.gt(freq, high_freq)  # freq < low_freq || freq > high_freq
    before_l = torch.lt(freq, l)  # freq < l
    before_h = torch.lt(freq, h)  # freq < h
    after_h = torch.ge(freq, h)  # freq >= h

    # order of operations matter here (since there is overlapping frequency regions)
    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h] = scale * freq[before_h]
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high_freq] = freq[outside_low_high_freq]

    return res


def vtln_warp_mel_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq,
                       vtln_warp_factor, mel_freq):
    r"""
    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        mel_freq (torch.Tensor): Given frequency in Mel

    Returns:
        torch.Tensor: ``mel_freq`` after vtln warp
    """
    return mel_scale(vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq,
                                    vtln_warp_factor, inverse_mel_scale(mel_freq)))


def get_mel_banks(num_bins, window_length_padded, sample_freq,
                  low_freq, high_freq, vtln_low, vtln_high, vtln_warp_factor):
    # type: (int, int, float, float, float, float, float)
    """
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The tuple consists of ``bins`` (which is
        melbank of size (``num_bins``, ``num_fft_bins``)) and ``center_freqs`` (which is
        center frequencies of bins of size (``num_bins``)).
    """
    assert num_bins > 3, 'Must have at least 3 mel bins'
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq), \
        ('Bad values in options: low-freq %f and high-freq %f vs. nyquist %f' % (low_freq, high_freq, nyquist))

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or ((low_freq < vtln_low < high_freq) and
                                       (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high)), \
        ('Bad values in options: vtln-low %f and vtln-high %f, versus low-freq %f and high-freq %f' %
            (vtln_low, vtln_high, low_freq, high_freq))

    bin = torch.arange(num_bins, dtype=torch.get_default_dtype()).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
        center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
        right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)

    center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins, dtype=torch.get_default_dtype())).unsqueeze(0)

    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    if vtln_warp_factor == 1.0:
        # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
        bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
    else:
        # warping can move the order of left_mel, center_mel, right_mel anywhere
        bins = torch.zeros_like(up_slope)
        up_idx = torch.gt(mel, left_mel) & torch.le(mel, center_mel)  # left_mel < mel <= center_mel
        down_idx = torch.gt(mel, center_mel) & torch.lt(mel, right_mel)  # center_mel < mel < right_mel
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]

    return bins, center_freqs


def fbank(
        waveform, blackman_coeff=0.42, channel=-1, dither=1.0, energy_floor=0.0,
        frame_length=25.0, frame_shift=10.0, high_freq=0.0, htk_compat=False, low_freq=20.0,
        min_duration=0.0, num_mel_bins=23, preemphasis_coefficient=0.97, raw_energy=True,
        remove_dc_offset=True, round_to_power_of_two=True, sample_frequency=16000.0,
        snip_edges=True, subtract_mean=False, use_energy=False, use_log_fbank=True, use_power=True,
        vtln_high=-500.0, vtln_low=100.0, vtln_warp=1.0, window_type=POVEY):
    r"""Create a fbank from a raw audio signal. This matches the input/output of Kaldi's
    compute-fbank-feats.

    Args:
        waveform (torch.Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        channel (int): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``1.0``)
        energy_floor (float): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``0.0``)
        frame_length (float): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float): Frame shift in milliseconds (Default: ``10.0``)
        high_freq (float): High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (Default: ``0.0``)
        htk_compat (bool): If true, put energy last.  Warning: not sufficient to get HTK compatible features (need
            to change other parameters). (Default: ``False``)
        low_freq (float): Low cutoff frequency for mel bins (Default: ``20.0``)
        min_duration (float): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        num_mel_bins (int): Number of triangular mel-frequency bins (Default: ``23``)
        preemphasis_coefficient (float): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset: Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        use_energy (bool): Add an extra dimension with energy to the FBANK output. (Default: ``False``)
        use_log_fbank (bool):If true, produce log-filterbank, else produce linear. (Default: ``True``)
        use_power (bool): If true, use power, else use magnitude. (Default: ``True``)
        vtln_high (float): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (Default: ``-500.0``)
        vtln_low (float): Low inflection point in piecewise linear VTLN warping function (Default: ``100.0``)
        vtln_warp (float): Vtln warp factor (only applicable if vtln_map not specified) (Default: ``1.0``)
        window_type (str): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman') (Default: ``'povey'``)

    Returns:
        torch.Tensor: A fbank identical to what Kaldi would output. The shape is (m, ``num_mel_bins + use_energy``)
        where m is calculated in _get_strided
    """
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0)

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
        snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient)

    # size (m, padded_window_size // 2 + 1, 2)
    fft = torch.rfft(strided_input, 1, normalized=False, onesided=True)

    power_spectrum = fft.pow(2).sum(2).unsqueeze(1)  # size (m, 1, padded_window_size // 2 + 1)
    if not use_power:
        power_spectrum = power_spectrum.pow(0.5)

    # size (num_mel_bins, padded_window_size // 2)
    mel_energies, _ = get_mel_banks(num_mel_bins, padded_window_size, sample_frequency,
                                    low_freq, high_freq, vtln_low, vtln_high, vtln_warp)

    # pad right column with zeros and add dimension, size (1, num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode='constant', value=0).unsqueeze(0)

    # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = (power_spectrum * mel_energies).sum(dim=2)
    if use_log_fbank:
        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, EPSILON).log()

    # if use_energy then add it as the last column for htk_compat == true else first column
    if use_energy:
        signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
        # returns size (m, num_mel_bins + 1)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
        else:
            mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)

    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies


def _get_dct_matrix(num_ceps, num_mel_bins):
    # returns a dct matrix of size (num_mel_bins, num_ceps)
    # size (num_mel_bins, num_mel_bins)
    dct_matrix = torchaudio.functional.create_dct(num_mel_bins, num_mel_bins, 'ortho')
    # kaldi expects the first cepstral to be weighted sum of factor sqrt(1/num_mel_bins)
    # this would be the first column in the dct_matrix for torchaudio as it expects a
    # right multiply (which would be the first column of the kaldi's dct_matrix as kaldi
    # expects a left multiply e.g. dct_matrix * vector).
    dct_matrix[:, 0] = math.sqrt(1 / float(num_mel_bins))
    dct_matrix = dct_matrix[:, :num_ceps]
    return dct_matrix


def _get_lifter_coeffs(num_ceps, cepstral_lifter):
    # returns size (num_ceps)
    # Compute liftering coefficients (scaling on cepstral coeffs)
    # coeffs are numbered slightly differently from HTK: the zeroth index is C0, which is not affected.
    i = torch.arange(num_ceps, dtype=torch.get_default_dtype())
    return 1.0 + 0.5 * cepstral_lifter * torch.sin(math.pi * i / cepstral_lifter)


def mfcc(
        waveform, blackman_coeff=0.42, cepstral_lifter=22.0, channel=-1, dither=1.0,
        energy_floor=0.0, frame_length=25.0, frame_shift=10.0, high_freq=0.0, htk_compat=False,
        low_freq=20.0, num_ceps=13, min_duration=0.0, num_mel_bins=23, preemphasis_coefficient=0.97,
        raw_energy=True, remove_dc_offset=True, round_to_power_of_two=True,
        sample_frequency=16000.0, snip_edges=True, subtract_mean=False, use_energy=False,
        vtln_high=-500.0, vtln_low=100.0, vtln_warp=1.0, window_type=POVEY):
    r"""Create a mfcc from a raw audio signal. This matches the input/output of Kaldi's
    compute-mfcc-feats.

    Args:
        waveform (torch.Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        cepstral_lifter (float): Constant that controls scaling of MFCCs (Default: ``22.0``)
        channel (int): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``1.0``)
        energy_floor (float): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``0.0``)
        frame_length (float): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float): Frame shift in milliseconds (Default: ``10.0``)
        high_freq (float): High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (Default: ``0.0``)
        htk_compat (bool): If true, put energy last.  Warning: not sufficient to get HTK compatible features (need
            to change other parameters). (Default: ``False``)
        low_freq (float): Low cutoff frequency for mel bins (Default: ``20.0``)
        num_ceps (int): Number of cepstra in MFCC computation (including C0) (Default: ``13``)
        min_duration (float): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        num_mel_bins (int): Number of triangular mel-frequency bins (Default: ``23``)
        preemphasis_coefficient (float): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset: Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        use_energy (bool): Add an extra dimension with energy to the FBANK output. (Default: ``False``)
        vtln_high (float): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (Default: ``-500.0``)
        vtln_low (float): Low inflection point in piecewise linear VTLN warping function (Default: ``100.0``)
        vtln_warp (float): Vtln warp factor (only applicable if vtln_map not specified) (Default: ``1.0``)
        window_type (str): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman') (Default: ``'povey'``)

    Returns:
        torch.Tensor: A mfcc identical to what Kaldi would output. The shape is (m, ``num_ceps``)
        where m is calculated in _get_strided
    """
    assert num_ceps <= num_mel_bins, 'num_ceps cannot be larger than num_mel_bins: %d vs %d' % (num_ceps, num_mel_bins)

    # The mel_energies should not be squared (use_power=True), not have mean subtracted
    # (subtract_mean=False), and use log (use_log_fbank=True).
    # size (m, num_mel_bins + use_energy)
    feature = fbank(waveform=waveform, blackman_coeff=blackman_coeff, channel=channel,
                    dither=dither, energy_floor=energy_floor, frame_length=frame_length,
                    frame_shift=frame_shift, high_freq=high_freq, htk_compat=htk_compat,
                    low_freq=low_freq, min_duration=min_duration, num_mel_bins=num_mel_bins,
                    preemphasis_coefficient=preemphasis_coefficient, raw_energy=raw_energy,
                    remove_dc_offset=remove_dc_offset, round_to_power_of_two=round_to_power_of_two,
                    sample_frequency=sample_frequency, snip_edges=snip_edges, subtract_mean=False,
                    use_energy=use_energy, use_log_fbank=True, use_power=True,
                    vtln_high=vtln_high, vtln_low=vtln_low, vtln_warp=vtln_warp, window_type=window_type)

    if use_energy:
        # size (m)
        signal_log_energy = feature[:, num_mel_bins if htk_compat else 0]
        # offset is 0 if htk_compat==True else 1
        mel_offset = int(not htk_compat)
        feature = feature[:, mel_offset:(num_mel_bins + mel_offset)]

    # size (num_mel_bins, num_ceps)
    dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins)

    # size (m, num_ceps)
    feature = feature.matmul(dct_matrix)

    if cepstral_lifter != 0.0:
        # size (1, num_ceps)
        lifter_coeffs = _get_lifter_coeffs(num_ceps, cepstral_lifter).unsqueeze(0)
        feature *= lifter_coeffs

    # if use_energy then replace the last column for htk_compat == true else first column
    if use_energy:
        feature[:, 0] = signal_log_energy

    if htk_compat:
        energy = feature[:, 0].unsqueeze(1)  # size (m, 1)
        feature = feature[:, 1:]  # size (m, num_ceps - 1)
        if not use_energy:
            # scale on C0 (actually removing a scale we previously added that's
            # part of one common definition of the cosine transform.)
            energy *= math.sqrt(2)

        feature = torch.cat((feature, energy), dim=1)

    feature = _subtract_column_mean(feature, subtract_mean)
    return feature


def _get_LR_indices_and_weights(orig_freq, new_freq, output_samples_in_unit, window_width,
                                lowpass_cutoff, lowpass_filter_width):
    r"""Based on LinearResample::SetIndexesAndWeights where it retrieves the weights for
    resampling as well as the indices in which they are valid. LinearResample (LR) means
    that the output signal is at linearly spaced intervals (i.e the output signal has a
    frequency of ``new_freq``). It uses sinc/bandlimited interpolation to upsample/downsample
    the signal.

    The reason why the same filter is not used for multiple convolutions is because the
    sinc function could sampled at different points in time. For example, suppose
    a signal is sampled at the timestamps (seconds)
    0         16        32
    and we want it to be sampled at the timestamps (seconds)
    0 5 10 15   20 25 30  35
    at the timestamp of 16, the delta timestamps are
    16 11 6 1   4  9  14  19
    at the timestamp of 32, the delta timestamps are
    32 27 22 17 12 8 2    3

    As we can see from deltas, the sinc function is sampled at different points of time
    assuming the center of the sinc function is at 0, 16, and 32 (the deltas [..., 6, 1, 4, ....]
    for 16 vs [...., 2, 3, ....] for 32)

    Example, one case is when the ``orig_freq`` and ``new_freq`` are multiples of each other then
    there needs to be one filter.

    A windowed filter function (i.e. Hanning * sinc) because the ideal case of sinc function
    has infinite support (non-zero for all values) so instead it is truncated and multiplied by
    a window function which gives it less-than-perfect rolloff [1].

    [1] Chapter 16: Windowed-Sinc Filters, https://www.dspguide.com/ch16/1.htm

    Args:
        orig_freq (float): The original frequency of the signal
        new_freq (float): The desired frequency
        output_samples_in_unit (int): The number of output samples in the smallest repeating unit:
            num_samp_out = new_freq / Gcd(orig_freq, new_freq)
        window_width (float): The width of the window which is nonzero
        lowpass_cutoff (float): The filter cutoff in Hz. The filter cutoff needs to be less
            than samp_rate_in_hz/2 and less than samp_rate_out_hz/2.
        lowpass_filter_width (int): Controls the sharpness of the filter, more == sharper but less
            efficient. We suggest around 4 to 10 for normal use

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of ``min_input_index`` (which is the minimum indices
        where the window is valid, size (``output_samples_in_unit``)) and ``weights`` (which is the weights
        which correspond with min_input_index, size (``output_samples_in_unit``, ``max_weight_width``)).
    """
    assert lowpass_cutoff < min(orig_freq, new_freq) / 2
    output_t = torch.arange(0, output_samples_in_unit, dtype=torch.get_default_dtype()) / new_freq
    min_t = output_t - window_width
    max_t = output_t + window_width

    min_input_index = torch.ceil(min_t * orig_freq)  # size (output_samples_in_unit)
    max_input_index = torch.floor(max_t * orig_freq)  # size (output_samples_in_unit)
    num_indices = max_input_index - min_input_index + 1  # size (output_samples_in_unit)

    max_weight_width = num_indices.max()
    # create a group of weights of size (output_samples_in_unit, max_weight_width)
    j = torch.arange(max_weight_width).unsqueeze(0)
    input_index = min_input_index.unsqueeze(1) + j
    delta_t = (input_index / orig_freq) - output_t.unsqueeze(1)

    weights = torch.zeros_like(delta_t)
    inside_window_indices = delta_t.abs().lt(window_width)
    # raised-cosine (Hanning) window with width `window_width`
    weights[inside_window_indices] = 0.5 * (1 + torch.cos(2 * math.pi * lowpass_cutoff /
                                            lowpass_filter_width * delta_t[inside_window_indices]))

    t_eq_zero_indices = delta_t.eq(0.0)
    t_not_eq_zero_indices = ~t_eq_zero_indices
    # sinc filter function
    weights[t_not_eq_zero_indices] *= torch.sin(
        2 * math.pi * lowpass_cutoff * delta_t[t_not_eq_zero_indices]) / (math.pi * delta_t[t_not_eq_zero_indices])
    # limit of the function at t = 0
    weights[t_eq_zero_indices] *= 2 * lowpass_cutoff

    weights /= orig_freq  # size (output_samples_in_unit, max_weight_width)
    return min_input_index, weights


def _lcm(a, b):
    return abs(a * b) // fractions.gcd(a, b)


def _get_num_LR_output_samples(input_num_samp, samp_rate_in, samp_rate_out):
    r"""Based on LinearResample::GetNumOutputSamples. LinearResample (LR) means that
    the output signal is at linearly spaced intervals (i.e the output signal has a
    frequency of ``new_freq``). It uses sinc/bandlimited interpolation to upsample/downsample
    the signal.

    Args:
        input_num_samp (int): The number of samples in the input
        samp_rate_in (float): The original frequency of the signal
        samp_rate_out (float): The desired frequency

    Returns:
        int: The number of output samples
    """
    # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
    # where tick_freq is the least common multiple of samp_rate_in and
    # samp_rate_out.
    samp_rate_in = int(samp_rate_in)
    samp_rate_out = int(samp_rate_out)

    tick_freq = _lcm(samp_rate_in, samp_rate_out)
    ticks_per_input_period = tick_freq // samp_rate_in

    # work out the number of ticks in the time interval
    # [ 0, input_num_samp/samp_rate_in ).
    interval_length_in_ticks = input_num_samp * ticks_per_input_period
    if interval_length_in_ticks <= 0:
        return 0
    ticks_per_output_period = tick_freq // samp_rate_out
    # Get the last output-sample in the closed interval, i.e. replacing [ ) with
    # [ ].  Note: integer division rounds down.  See
    # http://en.wikipedia.org/wiki/Interval_(mathematics) for an explanation of
    # the notation.
    last_output_samp = interval_length_in_ticks // ticks_per_output_period
    # We need the last output-sample in the open interval, so if it takes us to
    # the end of the interval exactly, subtract one.
    if last_output_samp * ticks_per_output_period == interval_length_in_ticks:
        last_output_samp -= 1
    # First output-sample index is zero, so the number of output samples
    # is the last output-sample plus one.
    num_output_samp = last_output_samp + 1
    return num_output_samp


def resample_waveform(waveform, orig_freq, new_freq, lowpass_filter_width=6):
    r"""Resamples the waveform at the new frequency. This matches Kaldi's OfflineFeatureTpl ResampleWaveform
    which uses a LinearResample (resample a signal at linearly spaced intervals to upsample/downsample
    a signal). LinearResample (LR) means that the output signal is at linearly spaced intervals (i.e
    the output signal has a frequency of ``new_freq``). It uses sinc/bandlimited interpolation to
    upsample/downsample the signal.

    https://ccrma.stanford.edu/~jos/resample/Theory_Ideal_Bandlimited_Interpolation.html
    https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

    Args:
        waveform (torch.Tensor): The input signal of size (c, n)
        orig_freq (float): The original frequency of the signal
        new_freq (float): The desired frequency
        lowpass_filter_width (int): Controls the sharpness of the filter, more == sharper
            but less efficient. We suggest around 4 to 10 for normal use. (Default: ``6``)

    Returns:
        torch.Tensor: The waveform at the new frequency
    """
    assert waveform.dim() == 2
    assert orig_freq > 0.0 and new_freq > 0.0

    min_freq = min(orig_freq, new_freq)
    lowpass_cutoff = 0.99 * 0.5 * min_freq

    assert lowpass_cutoff * 2 <= min_freq

    base_freq = fractions.gcd(int(orig_freq), int(new_freq))
    input_samples_in_unit = int(orig_freq) // base_freq
    output_samples_in_unit = int(new_freq) // base_freq

    window_width = lowpass_filter_width / (2.0 * lowpass_cutoff)
    first_indices, weights = _get_LR_indices_and_weights(orig_freq, new_freq, output_samples_in_unit,
                                                         window_width, lowpass_cutoff, lowpass_filter_width)
    weights = weights.to(waveform.device)

    assert first_indices.dim() == 1
    # TODO figure a better way to do this. conv1d reaches every element i*stride + padding
    # all the weights have the same stride but have different padding.
    # Current implementation takes the input and applies the various padding before
    # doing a conv1d for that specific weight.
    conv_stride = input_samples_in_unit
    conv_transpose_stride = output_samples_in_unit
    num_channels, wave_len = waveform.size()
    window_size = weights.size(1)
    tot_output_samp = _get_num_LR_output_samples(wave_len, orig_freq, new_freq)
    output = torch.zeros((num_channels, tot_output_samp),
                         device=waveform.device)
    # eye size: (num_channels, num_channels, 1)
    eye = torch.eye(num_channels, device=waveform.device).unsqueeze(2)
    for i in range(first_indices.size(0)):
        wave_to_conv = waveform
        first_index = int(first_indices[i].item())
        if first_index >= 0:
            # trim the signal as the filter will not be applied before the first_index
            wave_to_conv = wave_to_conv[..., first_index:]

        # pad the right of the signal to allow partial convolutions meaning compute
        # values for partial windows (e.g. end of the window is outside the signal length)
        max_unit_index = (tot_output_samp - 1) // output_samples_in_unit
        end_index_of_last_window = max_unit_index * conv_stride + window_size
        current_wave_len = wave_len - first_index
        right_padding = max(0, end_index_of_last_window + 1 - current_wave_len)

        left_padding = max(0, -first_index)
        if left_padding != 0 or right_padding != 0:
            wave_to_conv = torch.nn.functional.pad(wave_to_conv, (left_padding, right_padding))

        conv_wave = torch.nn.functional.conv1d(
            wave_to_conv.unsqueeze(0), weights[i].repeat(num_channels, 1, 1),
            stride=conv_stride, groups=num_channels)

        # we want conv_wave[:, i] to be at output[:, i + n*conv_transpose_stride]
        dilated_conv_wave = torch.nn.functional.conv_transpose1d(
            conv_wave, eye, stride=conv_transpose_stride).squeeze(0)

        # pad dilated_conv_wave so it reaches the output length if needed.
        dialated_conv_wave_len = dilated_conv_wave.size(-1)
        left_padding = i
        right_padding = max(0, tot_output_samp - (left_padding + dialated_conv_wave_len))
        dilated_conv_wave = torch.nn.functional.pad(
            dilated_conv_wave, (left_padding, right_padding))[..., :tot_output_samp]

        output += dilated_conv_wave

    return output
