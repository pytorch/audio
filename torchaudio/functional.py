import math
import torch


__all__ = [
    'scale',
    'pad_trim',
    'downmix_mono',
    'LC2CL',
    'istft',
    'spectrogram',
    'spectrogram_to_DB',
    'create_fb_matrix',
    'create_dct',
    'mu_law_encoding',
    'mu_law_expanding'
]


@torch.jit.script
def scale(tensor, factor):
    # type: (Tensor, float) -> Tensor
    r"""Scales tensor by a factor. By default, assuming the input is int32, it
    will scale the tensor to have values between -1.0 and 1.0.

    Args:
        tensor (torch.Tensor): Tensor input to scale
        factor (float): Factor to scale by

    Returns:
        torch.Tensor: Scaled by the scale factor
    """
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)

    return tensor / factor


@torch.jit.script
def pad_trim(waveform, max_len, fill_value):
    # type: (Tensor, int, float) -> Tensor
    r"""Pad/trim a 2D tensor

    Args:
        waveform (torch.Tensor): Tensor of audio of size (c, n)
        max_len (int): Length to which the waveform will be padded
        fill_value (float): Value to fill in

    Returns:
        torch.Tensor: Padded/trimmed tensor
    """
    n = waveform.size(1)
    if max_len > n:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (0, max_len - n), 'constant', fill_value)
    else:
        waveform = waveform[:, :max_len]
    return waveform


@torch.jit.script
def downmix_mono(waveform):
    # type: (Tensor) -> Tensor
    r"""Downmix stereo waveform to mono. Consider using a `SoxEffectsChain` with
    the `channels` effect instead of this transformation.

    Args:
        waveform (torch.Tensor): Tensor of audio of size (c, n)

    Returns:
        torch.Tensor: Tensor that has been downmixed of size (1, n)
    """
    if not waveform.is_floating_point():
        waveform = waveform.to(torch.float32)

    waveform = torch.mean(waveform, 0, True)
    return waveform


@torch.jit.script
def LC2CL(tensor):
    # type: (Tensor) -> Tensor
    r"""Permute a 2D tensor from samples (n, c) to (c, n).

    Args:
        tensor (torch.Tensor): Tensor of audio signal with shape (n, c)

    Returns:
        torch.Tensor: Tensor of audio signal with shape (c, n)
    """
    return tensor.transpose(0, 1).contiguous()


# TODO: remove this once https://github.com/pytorch/pytorch/issues/21478 gets solved
@torch.jit.ignore
def _stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided):
    # type: (Tensor, int, Optional[int], Optional[int], Optional[Tensor], bool, str, bool, bool) -> Tensor
    return torch.stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided)


def istft(stft_matrix,          # type: Tensor
          n_fft,                # type: int
          hop_length=None,      # type: Optional[int]
          win_length=None,      # type: Optional[int]
          window=None,          # type: Optional[Tensor]
          center=True,          # type: bool
          pad_mode='reflect',   # type: str
          normalized=False,     # type: bool
          onesided=True,        # type: bool
          length=None           # type: Optional[int]
          ):
    # type: (...) -> Tensor
    r""" Inverse short time Fourier Transform. This is expected to be the inverse of torch.stft.
    It has the same parameters (+ additional optional parameter of ``length``) and it should return the
    least squares estimation of the original signal. The algorithm will check using the NOLA condition (
    nonzero overlap).

    Important consideration in the parameters ``window`` and ``center`` so that the envelop
    created by the summation of all the windows is never zero at certain point in time. Specifically,
    :math:`\sum_{t=-\infty}^{\infty} w^2[n-t\times hop\_length] \cancel{=} 0`.

    Since stft discards elements at the end of the signal if they do not fit in a frame, the
    istft may return a shorter signal than the original signal (can occur if `center` is False
    since the signal isn't padded).

    If ``center`` is True, then there will be padding e.g. 'constant', 'reflect', etc. Left padding
    can be trimmed off exactly because they can be calculated but right padding cannot be calculated
    without additional information.

    Example: Suppose the last window is:
    [17, 18, 0, 0, 0] vs [18, 0, 0, 0, 0]

    The n_frames, hop_length, win_length are all the same which prevents the calculation of right padding.
    These additional values could be zeros or a reflection of the signal so providing ``length``
    could be useful. If ``length`` is ``None`` then padding will be aggressively removed
    (some loss of signal).

    [1] D. W. Griffin and J. S. Lim, “Signal estimation from modified short-time Fourier transform,”
    IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Args:
        stft_matrix (torch.Tensor): Output of stft where each row of a batch is a frequency and each
            column is a window. it has a shape of either (batch, fft_size, n_frames, 2) or (
            fft_size, n_frames, 2)
        n_fft (int): Size of Fourier transform
        hop_length (Optional[int]): The distance between neighboring sliding window frames.
            (Default: ``win_length // 4``)
        win_length (Optional[int]): The size of window frame and STFT filter. (Default: ``n_fft``)
        window (Optional[torch.Tensor]): The optional window function.
            (Default: ``torch.ones(win_length)``)
        center (bool): Whether ``input`` was padded on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`
        pad_mode (str): Controls the padding method used when ``center`` is ``True``
        normalized (bool): Whether the STFT was normalized
        onesided (bool): Whether the STFT is onesided
        length (Optional[int]): The amount to trim the signal by (i.e. the
            original signal length). (Default: whole signal)

    Returns:
        torch.Tensor: Least squares estimation of the original signal of size
        (batch, signal_length) or (signal_length)
    """
    stft_matrix_dim = stft_matrix.dim()
    assert 3 <= stft_matrix_dim <= 4, ('Incorrect stft dimension: %d' % (stft_matrix_dim))

    if stft_matrix_dim == 3:
        # add a batch dimension
        stft_matrix = stft_matrix.unsqueeze(0)

    device = stft_matrix.device
    fft_size = stft_matrix.size(1)
    assert (onesided and n_fft // 2 + 1 == fft_size) or (not onesided and n_fft == fft_size), (
        'one_sided implies that n_fft // 2 + 1 == fft_size and not one_sided implies n_fft == fft_size. '
        + 'Given values were onesided: %s, n_fft: %d, fft_size: %d' % ('True' if onesided else False, n_fft, fft_size))

    # use stft defaults for Optionals
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # There must be overlap
    assert 0 < hop_length <= win_length
    assert 0 < win_length <= n_fft

    if window is None:
        window = torch.ones(win_length)

    assert window.dim() == 1 and window.size(0) == win_length

    if win_length != n_fft:
        # center window with pad left and right zeros
        left = (n_fft - win_length) // 2
        window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        assert window.size(0) == n_fft
    # win_length and n_fft are synonymous from here on

    stft_matrix = stft_matrix.transpose(1, 2)  # size (batch, n_frames, fft_size, 2)
    stft_matrix = torch.irfft(stft_matrix, 1, normalized,
                              onesided, signal_sizes=(n_fft,))  # size (batch, n_frames, n_fft)

    assert stft_matrix.size(2) == n_fft
    n_frames = stft_matrix.size(1)

    ytmp = stft_matrix * window.view(1, 1, n_fft)  # size (batch, n_frames, n_fft)
    # each column of a batch is a frame which needs to be overlap added at the right place
    ytmp = ytmp.transpose(1, 2)  # size (batch, n_fft, n_frames)

    eye = torch.eye(n_fft, requires_grad=False,
                    device=device).unsqueeze(1)  # size (n_fft, 1, n_fft)

    # this does overlap add where the frames of ytmp are added such that the i'th frame of
    # ytmp is added starting at i*hop_length in the output
    y = torch.nn.functional.conv_transpose1d(
        ytmp, eye, stride=hop_length, padding=0)  # size (batch, 1, expected_signal_len)

    # do the same for the window function
    window_sq = window.pow(2).view(n_fft, 1).repeat((1, n_frames)).unsqueeze(0)  # size (1, n_fft, n_frames)
    window_envelop = torch.nn.functional.conv_transpose1d(
        window_sq, eye, stride=hop_length, padding=0)  # size (1, 1, expected_signal_len)

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    assert y.size(2) == expected_signal_len
    assert window_envelop.size(2) == expected_signal_len

    half_n_fft = n_fft // 2
    # we need to trim the front padding away if center
    start = half_n_fft if center else 0
    end = -half_n_fft if length is None else start + length

    y = y[:, :, start:end]
    window_envelop = window_envelop[:, :, start:end]

    # check NOLA non-zero overlap condition
    window_envelop_lowest = window_envelop.abs().min()
    assert window_envelop_lowest > 1e-11, ('window overlap add min: %f' % (window_envelop_lowest))

    y = (y / window_envelop).squeeze(1)  # size (batch, expected_signal_len)

    if stft_matrix_dim == 3:  # remove the batch dimension
        y = y.squeeze(0)
    return y


@torch.jit.script
def spectrogram(waveform, pad, window, n_fft, hop_length, win_length, power, normalized):
    # type: (Tensor, int, Tensor, int, int, int, int, bool) -> Tensor
    r"""Create a spectrogram from a raw audio signal.

    Args:
        waveform (torch.Tensor): Tensor of audio of size (c, n)
        pad (int): Two sided padding of signal
        window (torch.Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of fft
        hop_length (int): Length of hop between STFT windows
        win_length (int): Window size
        power (int): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
        normalized (bool): Whether to normalize by magnitude after stft

    Returns:
        torch.Tensor: Channels x frequency x time (c, f, t), where channels
        is unchanged, frequency is `n_fft // 2 + 1` where `n_fft` is the number of
        fourier bins, and time is the number of window hops (n_frames).
    """
    assert waveform.dim() == 2

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = _stft(waveform, n_fft, hop_length, win_length, window,
                   True, 'reflect', False, True)

    if normalized:
        spec_f /= window.pow(2).sum().sqrt()
    spec_f = spec_f.pow(power).sum(-1)  # get power of "complex" tensor
    return spec_f


@torch.jit.script
def spectrogram_to_DB(specgram, multiplier, amin, db_multiplier, top_db=None):
    # type: (Tensor, float, float, float, Optional[float]) -> Tensor
    r"""Turns a spectrogram from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        specgram (torch.Tensor): Normal STFT of size (c, f, t)
        multiplier (float): Use 10. for power and 20. for amplitude
        amin (float): Number to clamp specgram
        db_multiplier (float): Log10(max(reference value and amin))
        top_db (Optional[float]): Minimum negative cut-off in decibels. A reasonable number
            is 80.

    Returns:
        torch.Tensor: Spectrogram in DB of size (c, f, t)
    """
    specgram_db = multiplier * torch.log10(torch.clamp(specgram, min=amin))
    specgram_db -= multiplier * db_multiplier

    if top_db is not None:
        new_spec_db_max = torch.tensor(float(specgram_db.max()) - top_db,
                                       dtype=specgram_db.dtype, device=specgram_db.device)
        specgram_db = torch.max(specgram_db, new_spec_db_max)

    return specgram_db


@torch.jit.script
def create_fb_matrix(n_freqs, f_min, f_max, n_mels):
    # type: (int, float, float, int) -> Tensor
    r""" Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency
        f_max (float): Maximum frequency
        n_mels (int): Number of mel filterbanks

    Returns:
        torch.Tensor: Triangular filter banks (fb matrix) of size (`n_freqs`, `n_mels`)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., `n_freqs`), the applied result would be
        `A * create_fb_matrix(A.size(-1), ...)`.
    """
    # freq bins
    freqs = torch.linspace(f_min, f_max, n_freqs)
    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 0. if f_min == 0 else 2595. * math.log10(1. + (f_min / 700.))
    m_max = 2595. * math.log10(1. + (f_max / 700.))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700. * (10**(m_pts / 2595.) - 1.)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))
    return fb


@torch.jit.script
def create_dct(n_mfcc, n_mels, norm):
    # type: (int, int, Optional[str]) -> Tensor
    r"""Creates a DCT transformation matrix with shape (`n_mels`, `n_mfcc`),
    normalized depending on norm.

    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (Optional[str]): Norm to use (either 'ortho' or None)

    Returns:
        torch.Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (`n_mels`, `n_mfcc`).
    """
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == 'ortho'
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


@torch.jit.script
def mu_law_encoding(x, qc):
    # type: (Tensor, int) -> Tensor
    r"""Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1.

    Args:
        x (torch.Tensor): Input tensor
        qc (int): Number of channels (i.e. quantization channels)

    Returns:
        torch.Tensor: Input after mu-law companding
    """
    mu = qc - 1.
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype)
    x_mu = torch.sign(x) * torch.log1p(mu *
                                       torch.abs(x)) / torch.log1p(mu)
    x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu


@torch.jit.script
def mu_law_expanding(x_mu, qc):
    # type: (Tensor, int) -> Tensor
    r"""Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Args:
        x_mu (torch.Tensor): Input tensor
        qc (int): Number of channels (i.e. quantization channels)

    Returns:
        torch.Tensor: Input after decoding
    """
    mu = qc - 1.
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype)
    x = ((x_mu) / mu) * 2 - 1.
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
    return x
