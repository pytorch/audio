from __future__ import absolute_import, division, print_function, unicode_literals
import math
import torch
import torchaudio


__all__ = [
    'lowpass1',
]

# Should we name this low pass filter?
def lowpass1(waveform,           # type: Tensor
             sample_rate,        # type: int
             n_fft,              # type: int
             f_thresh,           # type: int
             hop_length=None,    # type: Optional[int]
             win_length=None,    # type: Optional[int]
             window=None,        # type: Optional[Tensor]
             center=True,        # type: Optional[bool]
             pad_mode='reflect', # type: Optional[str]
             normalized=False,   # type: Optional[bool]
             onesided=True,      # type: Optional[bool]
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

    assert(f_thresh < sample_rate // 2)
    assert(f_thresh >= 0)

    # Convert threshold frequency cutoff to index
    # Depends on the sampling rate
    i_thresh = math.ceil(f_thresh * n_fft / sample_rate)

    assert(i_thresh < n_fft // 2)

    waveform_stft = torch.stft( waveform,
                                n_fft,
                                hop_length,
                                win_length,
                                window,
                                center,
                                pad_mode,
                                normalized,
                                onesided
                               )


    # Zero out elements above the CUTOFF
    (n_channels, fft_size, n_windows, n_real_complex) = waveform_stft.size()
    assert(2 == n_real_complex)
    assert(1 == n_channels or 2 == n_channels)  # audio is mono or stereo
    
    # Set all components above threshold to zero
    i_thresh_end = fft_size if onesided else fft_size - i_thresh - 1
    waveform_stft[[slice(None), torch.arange(i_thresh, i_thresh_end).long(), slice(None), slice(None)]] = 0

    denoised_audio = torchaudio.functional.istft(waveform_stft,
                                                 n_fft,
                                                 hop_length,
                                                 win_length,
                                                 window,
                                                 center,
                                                 pad_mode,
                                                 normalized,
                                                 onesided,
                                                 waveform.size()[1])

    return denoised_audio
