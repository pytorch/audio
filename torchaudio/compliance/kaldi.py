import math
import random
import torch


__all__ = [
    'fbank',
    'spectrogram'
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
    """ Returns the log energy of size (m) for a strided_input (m,*)
    """
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
    if energy_floor == 0.0:
        return log_energy
    else:
        return torch.max(log_energy,
                         torch.tensor(math.log(energy_floor), dtype=torch.get_default_dtype()))


def _get_waveform_and_window_properties(sig, channel, sample_frequency, frame_shift,
                                        frame_length, round_to_power_of_two, preemphasis_coefficient):
    """Gets the waveform and window properties
    """
    waveform = sig[max(channel, 0), :]  # size (n)
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
    """Gets a window and it's log energy
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


def spectrogram(
        sig, blackman_coeff=0.42, channel=-1, dither=1.0, energy_floor=0.0,
        frame_length=25.0, frame_shift=10.0, min_duration=0.0,
        preemphasis_coefficient=0.97, raw_energy=True, remove_dc_offset=True,
        round_to_power_of_two=True, sample_frequency=16000.0, snip_edges=True,
        subtract_mean=False, window_type=POVEY):
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
            depends only on the frame_shift, and we reflect the data at the ends. (default = True)
        subtract_mean (bool): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (default = False)
        window_type (str): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman') (default = 'povey')

    Outputs:
        Tensor: a spectrogram identical to what Kaldi would output. The shape is (m, `padded_window_size` // 2 + 1)
            where m is calculated in _get_strided
    """
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        sig, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)

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

    if subtract_mean:
        col_means = torch.mean(power_spectrum, dim=0).unsqueeze(0)  # size (1, padded_window_size // 2 + 1)
        power_spectrum = power_spectrum - col_means

    return power_spectrum


def fbank(
        sig, blackman_coeff=0.42, channel=-1, debug_mel=False, dither=1.0, energy_floor=0.0,
        frame_length=25.0, frame_shift=10.0, high_freq=0.0, htk_compat=False, low_freq=20.0,
        min_duration=0.0, num_mel_bins=23, preemphasis_coefficient=0.97, raw_energy=True,
        remove_dc_offset=True, round_to_power_of_two=True, sample_frequency=16000.0,
        snip_edges=True, subtract_mean=False, use_energy=False, use_log_fbank=True, use_power=True,
        vtln_high=-500.0, vtln_low=100.0, vtln_warp=1.0, window_type='povey'):
    """Create a fbank from a raw audio signal. This matches the input/output of Kaldi's
    compute-fbank-feats.

    Inputs:
        sig (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float): Constant coefficient for generalized Blackman window. (default = 0.42)
        channel (int): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (default = -1)
        debug_mel (bool): Print out debugging information for mel bin computation (default = False)
        dither (float): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (default = 1.0)
        energy_floor (float): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (default = 0.0)
        frame_length (float): Frame length in milliseconds (default = 25.0)
        frame_shift (float): Frame shift in milliseconds (default = 10.0)
        high_freq (float): High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (default = 0.0)
        htk_compat (bool): If true, put energy last.  Warning: not sufficient to get HTK compatible features (need
            to change other parameters). (default = False)
        low_freq (float): Low cutoff frequency for mel bins (default = 20.0)
        min_duration (float): Minimum duration of segments to process (in seconds). (default = 0.0)
        num_mel_bins (int): Number of triangular mel-frequency bins (default = 23)
        preemphasis_coefficient (float): Coefficient for use in signal preemphasis (default = 0.97)
        raw_energy (bool): If True, compute energy before preemphasis and windowing (default = True)
        remove_dc_offset: Subtract mean from waveform on each frame (default = True)
        round_to_power_of_two (bool): If True, round window size to power of two by zero-padding input
            to FFT. (default = True)
        sample_frequency (float): Waveform data sample frequency (must match the waveform file, if
            specified there) (default = 16000.0)
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (default = True)
        subtract_mean (bool): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (default = False)
        use_energy (bool): Add an extra dimension with energy to the FBANK output. (default = False)
        use_log_fbank (bool):If true, produce log-filterbank, else produce linear. (default = True)
        use_power (bool): If true, use power, else use magnitude. (default = True)
        vtln_high (float): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (default = -500.0)
        vtln_low (float): Low inflection point in piecewise linear VTLN warping function (float, default = 100.0)
        vtln_warp (float): Vtln warp factor (only applicable if vtln_map not specified) (float, default = 1.0)
        window_type (str): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman') (default = 'povey')

    Outputs:
        Tensor: a fbank identical to what Kaldi would output. The shape is ...
    """
    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        sig, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient)

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0)

    strided_input, signal_log_energy = _get_window(
        waveform, padded_window_size, window_size, window_shift, window_type, blackman_coeff,
        snip_edges, raw_energy, energy_floor, dither, remove_dc_offset, preemphasis_coefficient)

    # size (m, padded_window_size // 2 + 1, 2)
    fft = torch.rfft(strided_input, 1, normalized=False, onesided=True)

    power_spectrum = fft.pow(2).sum(2)
    if not use_power:
        power_spectrum = power_spectrum.pow(0.5)

    mel_offset = 1 if use_energy and not htk_compat else 0
    # mel_energies = ()
    # power_spectrum = torch.max(, EPSILON).log()  # size (m, padded_window_size // 2 + 1)

    return torch.rand((2, 2))
