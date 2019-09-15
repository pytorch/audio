from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
import math
import torchaudio

"""
Biquad filtering approach based on SoX implementation 14.4.2

Performance needs to be evaluated against SoX

"""

__all__ = ["lowpass1", "biquad", "lowpass_biquad", "highpass_biquad"]


def _dB2Linear(x):
    return math.exp(x * math.log(10) / 20.0)


def highpass_biquad(input, sr, cutoff_freq, Q=0.707):

    GAIN = 1
    w0 = 2 * math.pi * cutoff_freq / sr
    A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    mult = _dB2Linear(max(GAIN, 0))

    b0 = (1 + math.cos(w0)) / 2
    b1 = -1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(input, b0, b1, b2, a0, a1, a2)


def lowpass_biquad(input, sr, cutoff_freq, Q=0.707):

    GAIN = 1
    w0 = 2 * math.pi * cutoff_freq / sr
    A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = math.sin(w0) / 2 / Q
    mult = _dB2Linear(max(GAIN, 0))

    b0 = (1 - math.cos(w0)) / 2
    b1 = 1 - math.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    return biquad(input, b0, b1, b2, a0, a1, a2)




def diff_eq_calculate(waveform, a_coeffs, b_coeffs):
    """
    waveform is the input audio tensor of size [n_channels, n_frames]

    expresses general difference equation of the form:

    a0 * o[n] + a1 * o[n-1] + a2 * o[n-2] + ... =
    b0 * i[n] + b1 * i[n-1] + b2 * i[n-2] + ...

    a_coeffs input list of length and [a0, a1, a2, ...]
    b_coeffs input list length and [b0, b1, b2, ...]

    """

    def clip(v):
        return max(-1.0, min(1.0, v))

    [n_channels, n_frames] = waveform.size()    
    n_a_coeffs = len(a_coeffs)
    n_b_coeffs = len(b_coeffs)
    n_order = max([n_a_coeffs, n_b_coeffs])
    assert(n_order > 0 and n_a_coeffs > 0)
    a0 = a_coeffs[0]

    coeff_matrix = torch.zeros(2, n_order, dtype=torch.float)
    # fill in the output coefficients
    for i_a, a in enumerate(a_coeffs):
        if i_a > 0:
            coeff_matrix[0, n_order - i_a - 1] = -1.0 * a / a0
    # fill in the input coefficients
    for i_b, b in enumerate(b_coeffs):
        coeff_matrix[1, n_order - i_b - 1] = float(b) / a0

    print("Coefficients matrix", coeff_matrix)

    # pad waveform by the order of the filter
    pad = torch.zeros(n_channels, n_order - 1)
    
    padded_waveform = torch.cat([pad, waveform], dim = 1)
    print("padded waveform", padded_waveform, padded_waveform.size())

    dat = torch.stack([torch.zeros_like(padded_waveform), padded_waveform])
    #print("padded data", dat.size(), dat)
    coeff_flattened = coeff_matrix.view(-1)
    for i in range(n_frames):
        for i_channel in range(n_channels):
            sub = dat[
                [
                    slice(None),
                    i_channel,
                    torch.arange(i, i + n_order).long(),
                ]
            ]
            dat[0, i_channel, i + n_order - 1] = clip(torch.dot(sub.view(-1), coeff_flattened))
    
    return dat[[0, slice(None), torch.arange(n_order - 1, n_frames + n_order-1)]]

def biquad(input, b0, b1, b2, a0, a1, a2):

    def clip(v):
        return max(-1.0, min(1.0, v))

    n_channels, n_samples = input.size()
    output = torch.zeros_like(input)

    # Normalize the coefficients
    b2 = float(b2) / a0
    b1 = float(b1) / a0
    b0 = float(b0) / a0
    a2 = float(a2) / a0
    a1 = float(a1) / a0

    # Initial Conditions set to zero
    pi1 = torch.zeros(n_channels, 1)
    pi2 = torch.zeros(n_channels, 1)
    po1 = torch.zeros(n_channels, 1)
    po2 = torch.zeros(n_channels, 1)

    for i in range(input.size()[1]):
        o0 = clip(input[:, i] * b0 + pi1 * b1 + pi2 * b2 - po1 * a1 - po2 * a2)
        pi2 = pi1
        pi1 = input[:, i]
        po2 = po1
        po1 = o0
        output[:, i] = o0

    return output

def lowpass1(
    waveform,  # type: Tensor
    sample_rate,  # type: int
    n_fft,  # type: int
    f_thresh,  # type: int
    hop_length=None,  # type: Optional[int]
    win_length=None,  # type: Optional[int]
    window=None,  # type: Optional[Tensor]
    center=True,  # type: Optional[bool]
    pad_mode="reflect",  # type: Optional[str]
    normalized=False,  # type: Optional[bool]
    onesided=True,  # type: Optional[bool]
):
    # type: (...) -> Tensor
    r"""Simple low pass filter.  Performs Short Time Fourier Transform, zeros out elements
        above a threshold requency, and inverts Fourier Transform.
        
    Args:
        waveform (torch.Tensor): Audio waveform that has a size of (channel, n_frames)
        sample_rate (int): Audio waveform sampling rate in frames / sec (Hz)
        n_fft (int): Size of Fourier transform in frames
        f_thresh (int): Threshold above which signal will be filtered out (Hz)
        hop_length (Optional[int]): See STFT documentation
        win_length (Optional[int]): See STFT documentation
        window (Optional[torch.Tensor]): See STFT documentation
        center (Optional[bool]): See STFT documentation
        pad_mode (Optional[str]): See STFT documentation
        normalized (Optional[bool]): See STFT documentation
        onesided (Optional[bool]): See STFT documentation

    Returns:
        torch.Tensor: Audio waveform that has been denoised of size (channel, n_frames)
    """

    assert f_thresh < sample_rate // 2
    assert f_thresh >= 0

    # Convert threshold frequency cutoff to index
    # Depends on the sampling rate
    i_thresh = math.ceil(f_thresh * n_fft / sample_rate)

    assert i_thresh < n_fft // 2

    waveform_stft = torch.stft(
        waveform,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
    )

    # Zero out elements above the CUTOFF
    (n_channels, fft_size, n_windows, n_real_complex) = waveform_stft.size()
    assert 2 == n_real_complex
    assert 1 == n_channels or 2 == n_channels  # audio is mono or stereo

    # Set all components above threshold to zero
    i_thresh_end = fft_size if onesided else fft_size - i_thresh - 1
    waveform_stft[
        [
            slice(None),
            torch.arange(i_thresh, i_thresh_end).long(),
            slice(None),
            slice(None),
        ]
    ] = 0

    denoised_audio = torchaudio.functional.istft(
        waveform_stft,
        n_fft,
        hop_length,
        win_length,
        window,
        center,
        pad_mode,
        normalized,
        onesided,
        waveform.size()[1],
    )

    return denoised_audio


if __name__ == '__main__':

    import _torch_filtering
    x = torch.rand(1, 8000 * 10, dtype=torch.float)

    
    b0 = 0.3
    b1 = 0.2
    b2 = 0.4
    a0 = 1.0
    a1 = 0.2
    a2 = 0.4
    print(_torch_filtering.biquad(x, x, a0, a1, a2, b0,b1, b2))
    

    """startdiff = time.time()
    v = diff_eq_calculate(x, [a0, a1, a2], [b0, b1, b2])
    enddiff = time.time()
    y = biquad(x, b0, b1, b2, a0, a1, a2)
    end2 = time.time()
    print(y.size())
    print(v.size())
    print(v, y)
    print("time1", enddiff - startdiff)
    print("time2", end2 - enddiff)"""