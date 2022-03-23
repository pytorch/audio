import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .feature_window import FrameExtractionOptions


def mel_scale(freq: float):
    return 1127.0 * math.log(1.0 + freq / 700.0)


def inverse_mel_scale(mel_freq: float):
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


# https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.cc#L150
def vtln_warp_freq(
    vtln_low_cutoff: float,  # upper+lower frequency cutoffs for VTLN.
    vtln_high_cutoff: float,
    low_freq: float,  # upper+lower frequency cutoffs in mel computation
    high_freq: float,
    vtln_warp_factor: float,
    freq: float,
):
    if not low_freq <= freq <= high_freq:
        return freq

    if vtln_low_cutoff <= low_freq:
        raise ValueError("'vtln_low_cutoff' must be larger than 'low_freq'")
    if high_freq <= vtln_high_cutoff:
        raise ValueError("'vtln_high_cutoff' must be smaller than 'high_freq'")

    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l
    Fh = scale * h

    assert l > low_freq and h < high_freq

    scale_left = (Fl - low_freq) / (l - low_freq)
    scale_right = (high_freq - Fh) / (high_freq - h)

    if freq < l:
        return low_freq + scale_left * (freq - low_freq)
    if freq < h:
        return scale * freq
    return high_freq + scale_right * (freq - high_freq)


# https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.cc#L213
def vtln_warp_mel_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float,
    mel_freq: float,
):
    # https://kaldi-asr.org/doc/mel-computations_8cc_source.html#l00213
    return mel_scale(
        vtln_warp_freq(
            vtln_low_cutoff,
            vtln_high_cutoff,
            low_freq,
            high_freq,
            vtln_warp_factor,
            inverse_mel_scale(mel_freq),
        ),
    )


@dataclass
class MelBanksOptions:
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.h#L43
    #
    # Main difference:
    # - Got rid of 'debug_mel' and 'htk_mode'
    num_bins: int
    low_freq: float = 20.0
    high_freq: float = 0.0
    vtln_low: float = 100.0
    vtln_high: float = -500.0


# https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.h#L78
#
# Main difference:
# - Change default `vtln_warp_factor` value to None.
class MelBanks:
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.cc#L33
    def __init__(
        self,
        opts: MelBanksOptions,
        frame_opts: FrameExtractionOptions,
        vtln_warp_factor: Optional[float] = None,
    ):
        num_bins = opts.num_bins
        if num_bins < 3:
            raise ValueError("Mel frequency bins have to be at least 3.")

        sample_freq = frame_opts.samp_freq
        window_length_padded = frame_opts.padded_window_size

        if window_length_padded % 2 != 0:
            raise ValueError(
                "Padded window length must be even. "
                "Tweak 'frame_length_ms' and 'sample_freq' to adjust the window length."
            )

        num_fft_bins = window_length_padded // 2
        nyquist = 0.5 * sample_freq
        low_freq = opts.low_freq
        high_freq = opts.high_freq if opts.high_freq > 0.0 else nyquist + opts.high_freq

        # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.cc#L51
        if not (0.0 <= low_freq < nyquist):
            raise ValueError(
                "The value of 'low_freq' must be non-negative and "
                f"less than nyquist frequency ({nyquist}). Found: {low_freq}."
            )
        if not (low_freq < high_freq <= nyquist):
            raise ValueError(
                f"The value of 'high_freq' must be greater than low_freq ({low_freq}) and "
                f"less than or equal to nyquist frequency ({nyquist}). Found: {high_freq}"
            )

        fft_bin_width = sample_freq / window_length_padded

        mel_low_freq = mel_scale(low_freq)
        mel_high_freq = mel_scale(high_freq)

        mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

        vtln_low = opts.vtln_low
        vtln_high = opts.vtln_high if opts.vtln_high > 0.0 else nyquist + opts.vtln_high

        # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.cc#L76
        if vtln_warp_factor is not None:
            if not (low_freq < vtln_low < high_freq):
                raise ValueError(
                    "When 'vtln_warp_factor' is provided, the value of 'vtln_low' must be "
                    f"within the range of 'low_freq' ({low_freq}) and 'high_freq' ({high_freq})."
                    f"Found: {vtln_low}."
                )
            if not (low_freq < vtln_high < high_freq):
                raise ValueError(
                    "When 'vtln_warp_factor' is provided, the value of 'vtln_high' must be "
                    f"with in the range of 'low_freq' ({low_freq}) and 'high_freq' ({high_freq})."
                    f"Found: {vtln_high}."
                )
            if not vtln_low < vtln_high:
                raise ValueError(
                    f"When 'vtln_warp_factor' is provided, the value of 'vtln_high' ({vtln_high}) "
                    f"must be greater than that of 'vtln_low' ({vtln_low})."
                )

        # TODO: vectorize this
        self._bins = []
        self._center_freqs = torch.empty(num_bins)
        # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.cc#L89
        for bin in range(num_bins):
            left_mel = mel_low_freq + bin * mel_freq_delta
            center_mel = mel_low_freq + (bin + 1) * mel_freq_delta
            right_mel = mel_low_freq + (bin + 2) * mel_freq_delta

            if vtln_warp_factor is not None:
                left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
                center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
                right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)

            self._center_freqs[bin] = inverse_mel_scale(center_mel)

            # https://kaldi-asr.org/doc/mel-computations_8cc_source.html#l00103
            this_bin = torch.zeros(num_fft_bins)
            first_index = -1
            last_index = -1
            for i in range(num_fft_bins):
                freq = i * fft_bin_width
                mel = mel_scale(freq)
                if left_mel < mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    this_bin[i] = weight
                    if first_index == -1:
                        first_index = i
                    last_index = i
            if first_index == -1 or last_index < first_index:
                raise ValueError(
                    "A bin in mel-scale conversion matrix is all zeros. "
                    "There might be too many mel bins. "
                    "Reducing the number of mel bins might help."
                )

            self._bins.append(this_bin)
            # print(f"bin offset = {first_index}, vec={this_bin}")
        self._bins = torch.stack(self._bins)

    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/mel-computations.cc#L226
    def compute(self, power_spectrum: Tensor) -> Tensor:
        # Note:
        # Probably better to redefine self._bins as the following.
        # Needs to update the other Computer code (fbank and mfcc)
        # print('mel compute 1:', self._bins[:, 1:])
        # print('mel compute 2:', power_spectrum[:, 1:-1].T)
        return torch.matmul(power_spectrum[:, 1:-1], self._bins[:, 1:].T)


def compute_lifter_coeffs(q: float, n: int) -> Tensor:
    coeffs = torch.empty(n)
    for i in range(n):
        coeffs[i] = 1.0 + 0.5 * q * math.sin(math.pi * i / q)
    return coeffs
