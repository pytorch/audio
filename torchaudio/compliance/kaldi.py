import math
import random
import torch


__all__ = [
    'spectrogram'
]


def _next_power_of_2(x):
    """ Returns the smallest power of 2 that is greater than x
    """
    return 1 if x == 0 else 2**(x - 1).bit_length()


def _get_strided(waveform, window_size, window_shift, snip_edges):
    """ Given a waveform (1D tensor of size num_samples), it returns a 2D tensor (m, window_size)
    representing how the window is shifted along the waveform. Each row is a frame.

    Inputs:
        sig (Tensor): Tensor of size num_samples
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.

    Output:
        Tensor: 2D tensor of size (m, window_size) where each row is a frame
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
    """ Returns a window function with the given type and size
    """
    if window_type == 'rectangular':
        return torch.ones(window_size, dtype=torch.get_default_dtype())

    a = 2 * math.pi / (window_size - 1)
    window_function = torch.arange(window_size, dtype=torch.get_default_dtype())
    if window_type == 'hanning':
        return 0.5 - 0.5 * torch.cos(a * window_function)
    elif window_type == 'hamming':
        return 0.54 - 0.46 * torch.cos(a * window_function)
    elif window_type == 'povey':
        # like hamming but goes to zero at edges
        return (0.5 - 0.5 * torch.cos(a * window_function)).pow(0.85)
    elif window_type == 'blackman':
        return blackman_coeff - 0.5 * torch.cos(a * window_function) + \
            (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
    else:
        raise Exception('Invalid window type ' + window_type)


def _get_log_energy(strided_input, epsilon, energy_floor):
    """ Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
    if energy_floor == 0.0:
        return log_energy
    else:
        return torch.max(log_energy,
                         torch.tensor(math.log(energy_floor), dtype=torch.get_default_dtype()))


def spectrogram(
        sig, allow_downsample=False, allow_upsample=False,
        blackman_coeff=0.42, channel=-1, dither=1.0, energy_floor=0.0, frame_length=25.0,
        frame_shift=10.0, max_feature_vectors=-1, min_duration=0.0, output_format='kaldi',
        preemphasis_coefficient=0.97, raw_energy=True, remove_dc_offset=True,
        round_to_power_of_two=True, sample_frequency=16000.0, snip_edges=True,
        subtract_mean=False, window_type='povey'):
    """Create a spectrogram from a raw audio signal. This matches the input/output of Kaldi's
    compute-spectrogram-feats.

    Inputs:
        sig (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float): Constant coefficient for generalized Blackman window. (default = 0.42)
        channel (int): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (default = -1)
        dither (float): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (default = 1.0)
        energy_floor (float): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (default = 0.0)
        frame_length (float): Frame length in milliseconds (default = 25.0)
        frame_shift (float): Frame shift in milliseconds (default = 10.0)
        min_duration (float): Minimum duration of segments to process (in seconds). (default = 0.0)
        preemphasis_coefficient (float): Coefficient for use in signal preemphasis (default = 0.97)
        raw_energy (bool): If True, compute energy before preemphasis and windowing (default = True)
        remove_dc_offset: Subtract mean from waveform on each frame (default = True)
        round_to_power_of_two (bool): If True, round window size to power of two by zero-padding input
            to FFT. (default = True)
        sample_frequency (float): Waveform data sample frequency (must match the waveform file, if
            specified there) (default = 16000.0)
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (default = true)
        subtract_mean (bool): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (default = False)
        window_type (str): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman') (default = 'povey')

    Not used Inputs:
        allow_downsample (bool): If True, allow the input waveform to have a higher frequency than the specified
            `sample_frequency` (and we'll downsample). (default = False)
        allow_upsample (bool): If True, allow the input waveform to have a lower frequency than the specified
            `sample_frequency` (and we'll upsample). (default = False)
        max_feature_vectors (int): Memory optimization. If larger than 0, periodically remove feature vectors so
            that only this number of the latest feature vectors is retained. (default = -1)
        output_format (str): Format of the output files [kaldi, htk] (default = 'kaldi')

    Outputs:
        Tensor: a spectrogram identical to what Kaldi would output. The shape is (, `padded_window_size` // 2 + 1)
    """
    # TODO figure out how numeric_limits<float>::epsilon() is implemented
    epsilon = torch.tensor(1.19209e-07, dtype=torch.get_default_dtype())

    waveform = sig[max(channel, 0), :]  # size (n)
    window_shift = int(sample_frequency * 0.001 * frame_shift)
    window_size = int(sample_frequency * 0.001 * frame_length)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0)

    assert 2 <= window_size <= len(waveform), ('choose a window size %d that is [2, %d]' % (window_size, len(waveform)))
    assert 0 < window_shift, '`window_shift` must be greater than 0'
    assert padded_window_size % 2 == 0, 'the padded ' \
        '`window_size` must be divisible by two. use `round_to_power_of_two` or change `frame_length`'
    assert 0. <= preemphasis_coefficient <= 1.0, '`preemphasis_coefficient` must be between [0,1]'

    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        # Returns a random number strictly between 0 and 1
        x = torch.max(epsilon, torch.rand(strided_input.shape))
        rand_gauss = torch.sqrt(-2 * x.log()) * torch.cos(2 * math.pi * x)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        # Compute the log energy of each row/frame before applying preemphasis and
        # window function
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)

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
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)

    # size (m, padded_window_size // 2 + 1, 2)
    fft = torch.rfft(strided_input, 1, normalized=False, onesided=True)

    # Convert the FFT into a power spectrum
    power_spectrum = torch.max(fft.pow(2).sum(2), epsilon).log()  # size (m, padded_window_size // 2 + 1)
    power_spectrum[:, 0] = signal_log_energy

    if subtract_mean:
        col_means = torch.mean(power_spectrum, dim=0).unsqueeze(0)  # size (1, padded_window_size // 2 + 1)
        power_spectrum = power_spectrum - col_means

    return power_spectrum
