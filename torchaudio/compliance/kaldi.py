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
        window_type (str): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackmann') (default = 'povey')

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
    epsilon = 1e-7
    waveform = sig[max(channel, 0), :]  # size (n)
    window_shift = int(sample_frequency * 0.001 * frame_shift)
    window_size = int(sample_frequency * 0.001 * frame_length)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert len(waveform) >= min_duration * sample_frequency, 'signal is too short'
    assert window_size <= len(waveform), 'window size is too long. decrease window size'
    assert padded_window_size % 2 == 0, 'the padded ' \
        'window size must be divisible by two. use `round_to_power_of_two` or change `frame_length`'

    n_fft = padded_window_size
    hop_length = window_shift
    win_length = padded_window_size
    return None
    # window =
    # normalized = False
    # onesided = True
    # waveform.stft(n_fft, hop_length, win_length, window, normalized, onesided)
    # return sig
