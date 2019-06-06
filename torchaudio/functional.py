import math
import torch


__all__ = [
    'scale',
    'pad_trim',
    'downmix_mono',
    'LC2CL',
    'spectrogram',
    'create_fb_matrix',
    'spectrogram_to_DB',
    'create_dct',
    'BLC2CBL',
    'mu_law_encoding',
    'mu_law_expanding'
]


@torch.jit.script
def scale(tensor, factor):
    # type: (Tensor, int) -> Tensor
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".

    Inputs:
        tensor (Tensor): Tensor of audio of size (Samples x Channels)
        factor (int): Maximum value of input tensor

    Outputs:
        Tensor: Scaled by the scale factor
    """
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)

    return tensor / factor


@torch.jit.script
def pad_trim(tensor, ch_dim, max_len, len_dim, fill_value):
    # type: (Tensor, int, int, int, float) -> Tensor
    """Pad/Trim a 2d-Tensor (Signal or Labels)

    Inputs:
        tensor (Tensor): Tensor of audio of size (n x c) or (c x n)
        ch_dim (int): Dimension of channel (not size)
        max_len (int): Length to which the tensor will be padded
        len_dim (int): Dimension of length (not size)
        fill_value (float): Value to fill in

    Outputs:
        Tensor: Padded/trimmed tensor
    """
    if max_len > tensor.size(len_dim):
        # array of [padding_left, padding_right, padding_top, padding_bottom]
        # so pad similar to append (aka only right/bottom) and do not pad
        # the length dimension. assumes equal sizes of padding.
        padding = [max_len - tensor.size(len_dim)
                   if (i % 2 == 1) and (i // 2 != len_dim)
                   else 0
                   for i in [0, 1, 2, 3]]
        # TODO add "with torch.no_grad():" back when JIT supports it
        tensor = torch.nn.functional.pad(tensor, padding, "constant", fill_value)
    elif max_len < tensor.size(len_dim):
        tensor = tensor.narrow(len_dim, 0, max_len)
    return tensor


@torch.jit.script
def downmix_mono(tensor, ch_dim):
    # type: (Tensor, int) -> Tensor
    """Downmix any stereo signals to mono.

    Inputs:
        tensor (Tensor): Tensor of audio of size (c x n) or (n x c)
        ch_dim (int): Dimension of channel (not size)

    Outputs:
        Tensor: Mono signal
    """
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)

    tensor = torch.mean(tensor, ch_dim, True)
    return tensor


@torch.jit.script
def LC2CL(tensor):
    # type: (Tensor) -> Tensor
    """Permute a 2d tensor from samples (n x c) to (c x n)

    Inputs:
        tensor (Tensor): Tensor of audio signal with shape (LxC)

    Outputs:
        Tensor: Tensor of audio signal with shape (CxL)
    """
    return tensor.transpose(0, 1).contiguous()


def _stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided):
    # type: (Tensor, int, Optional[int], Optional[int], Optional[Tensor], bool, str, bool, bool) -> Tensor
    return torch.stft(input, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided)


@torch.jit.script
def spectrogram(sig, pad, window, n_fft, hop, ws, power, normalize):
    # type: (Tensor, int, Tensor, int, int, int, int, bool) -> Tensor
    """Create a spectrogram from a raw audio signal

    Inputs:
        sig (Tensor): Tensor of audio of size (c, n)
        pad (int): two sided padding of signal
        window (Tensor): window_tensor
        n_fft (int): size of fft
        hop (int): length of hop between STFT windows
        ws (int): window size
        power (int > 0 ) : Exponent for the magnitude spectrogram,
                        e.g., 1 for energy, 2 for power, etc.
        normalize (bool) : whether to normalize by magnitude after stft


    Outputs:
        Tensor: channels x hops x n_fft (c, l, f), where channels
            is unchanged, hops is the number of hops, and n_fft is the
            number of fourier bins, which should be the window size divided
            by 2 plus 1.
    """
    assert sig.dim() == 2

    if pad > 0:
        # TODO add "with torch.no_grad():" back when JIT supports it
        sig = torch.nn.functional.pad(sig, (pad, pad), "constant")

    # default values are consistent with librosa.core.spectrum._spectrogram
    spec_f = _stft(sig, n_fft, hop, ws, window,
                   True, 'reflect', False, True).transpose(1, 2)

    if normalize:
        spec_f /= window.pow(2).sum().sqrt()
    spec_f = spec_f.pow(power).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
    return spec_f


@torch.jit.script
def create_fb_matrix(n_stft, f_min, f_max, n_mels):
    # type: (int, float, float, int) -> Tensor
    """ Create a frequency bin conversion matrix.

    Inputs:
        n_stft (int): number of filter banks from spectrogram
        f_min (float): minimum frequency
        f_max (float): maximum frequency
        n_mels (int): number of mel bins

    Outputs:
        Tensor: triangular filter banks (fb matrix)

    """
    # get stft freq bins
    stft_freqs = torch.linspace(f_min, f_max, n_stft)
    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 0. if f_min == 0 else 2595. * math.log10(1. + (f_min / 700.))
    m_max = 2595. * math.log10(1. + (f_max / 700.))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700. * (10**(m_pts / 2595.) - 1.)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - stft_freqs.unsqueeze(1)  # (n_stft, n_mels + 2)
    # create overlapping triangles
    z = torch.zeros(1)
    down_slopes = (-1. * slopes[:, :-2]) / f_diff[:-1]  # (n_stft, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_stft, n_mels)
    fb = torch.max(z, torch.min(down_slopes, up_slopes))
    return fb


@torch.jit.script
def spectrogram_to_DB(spec, multiplier, amin, db_multiplier, top_db=None):
    # type: (Tensor, float, float, float, Optional[float]) -> Tensor
    """Turns a spectrogram from the power/amplitude scale to the decibel scale.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Inputs:
        spec (Tensor): normal STFT
        multiplier (float): use 10. for power and 20. for amplitude
        amin (float): number to clamp spec
        db_multiplier (float): log10(max(reference value and amin))
        top_db (Optional[float]): minimum negative cut-off in decibels.  A reasonable number
            is 80.

    Outputs:
        Tensor: spectrogram in DB
    """
    spec_db = multiplier * torch.log10(torch.clamp(spec, min=amin))
    spec_db -= multiplier * db_multiplier

    if top_db is not None:
        new_spec_db_max = torch.tensor(float(spec_db.max()) - top_db, dtype=spec_db.dtype, device=spec_db.device)
        spec_db = torch.max(spec_db, new_spec_db_max)

    return spec_db


@torch.jit.script
def create_dct(n_mfcc, n_mels, norm):
    # type: (int, int, Optional[str]) -> Tensor
    """
    Creates a DCT transformation matrix with shape (num_mels, num_mfcc),
    normalized depending on norm

    Inputs:
        n_mfcc (int) : number of mfc coefficients to retain
        n_mels (int): number of MEL bins
        norm (Optional[str]) : norm to use (either 'ortho' or None)

    Outputs:
        Tensor: The transformation matrix, to be right-multiplied to row-wise data.
    """
    outdim = n_mfcc
    dim = n_mels
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(dim)
    k = torch.arange(outdim)[:, None]
    dct = torch.cos(math.pi / float(dim) * (n + 0.5) * k)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == 'ortho'
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(dim))
    return dct.t()


@torch.jit.script
def BLC2CBL(tensor):
    # type: (Tensor) -> Tensor
    """Permute a 3d tensor from Bands x Sample length x Channels to Channels x
       Bands x Samples length

    Inputs:
        tensor (Tensor): Tensor of spectrogram with shape (BxLxC)

    Outputs:
        Tensor: Tensor of spectrogram with shape (CxBxL)
    """
    return tensor.permute(2, 0, 1).contiguous()


@torch.jit.script
def mu_law_encoding(x, qc):
    # type: (Tensor, int) -> Tensor
    """Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1

    Inputs:
        x (Tensor): Input tensor
        qc (int): Number of channels (i.e. quantization channels)

    Outputs:
        Tensor: Input after mu-law companding
    """
    assert isinstance(x, torch.Tensor), 'mu_law_encoding expects a Tensor'
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
    """Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_

    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.

    Inputs:
        x_mu (Tensor): Input tensor
        qc (int): Number of channels (i.e. quantization channels)

    Outputs:
        Tensor: Input after decoding
    """
    assert isinstance(x_mu, torch.Tensor), 'mu_law_expanding expects a Tensor'
    mu = qc - 1.
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype)
    x = ((x_mu) / mu) * 2 - 1.
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
    return x
