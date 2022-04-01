"""Simplified port of

https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-window.h

"""
import math
from typing import Tuple

import torch
from torch import Tensor

from .math import round_up_to_nearest_power_of_two


class FrameExtractionOptions(torch.nn.Module):
    # Based off of
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-window.h#L35
    def __init__(
            self,
            samp_freq: float,
            frame_shift_ms: float = 10.0,
            frame_length_ms: float = 25.0,
            dither: float = 1.0,  # TODO: revise this. IIRC: Kaldi uses unnormalized data range
            preemph_coeff: float = 0.97,
            remove_dc_offset: bool = True,
            window_type: str = "povey",
            round_to_power_of_two: bool = True,
            blackman_coeff: float = 0.42,
            snip_edges: bool = True,
    ):
        super().__init__()

        if samp_freq <= 0:
            raise ValueError(f"'samp_freq' must be positive. Found: {samp_freq}")

        if frame_shift_ms <= 0:
            raise ValueError(f"'frame_shift_ms' must be positive. Found: {frame_shift_ms}")

        if frame_length_ms <= 0:
            raise ValueError(f"'frame_length_ms' must be positive. Found: {frame_length_ms}")

        if not 0.0 <= preemph_coeff <= 1.0:
            raise ValueError(f"'preemph_coeff' must be in [0.0, 1.0]. Found: {preemph_coeff}")

        self.samp_freq : float = samp_freq
        self.frame_shift_ms : float = frame_shift_ms
        self.frame_length_ms : float = frame_length_ms
        self.dither : float = dither
        self.preemph_coeff : float = preemph_coeff
        self.remove_dc_offset : bool = remove_dc_offset
        self.window_type : str = window_type
        self.round_to_power_of_two : bool = round_to_power_of_two
        self.blackman_coeff : float = blackman_coeff
        self.snip_edges : bool = snip_edges

    @property
    def window_shift(self) -> int:
        return int(self.samp_freq * 0.001 * self.frame_shift_ms)

    @property
    def window_size(self) -> int:
        return int(self.samp_freq * 0.001 * self.frame_length_ms)

    @property
    def padded_window_size(self):
        if self.round_to_power_of_two:
            return round_up_to_nearest_power_of_two(self.window_size)
        return self.window_size


def _get_window(
    window_type: str,
    window_size: int,
    blackman_coeff: float,
):
    device, dtype = torch.device("cpu"), torch.float64

    if window_type == "hanning":
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype)

    if window_type == "hamming":
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, device=device, dtype=dtype)

    if window_type == "povey":
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)

    if window_type == "rectangular":
        return torch.ones(window_size, device=device, dtype=dtype)

    if window_type == "blackman":
        a = 2 * math.pi / (window_size - 1)
        window = torch.arange(window_size, device=device, dtype=dtype)
        # can't use torch.blackman_window as they use different coefficients
        return blackman_coeff - 0.5 * torch.cos(a * window) + (0.5 - blackman_coeff) * torch.cos(2 * a * window)

    raise Exception("Invalid window type " + window_type)


def get_first_sample_of_frame(frame: int, opts: FrameExtractionOptions):
    frame_shift = opts.window_shift
    if opts.snip_edges:
        return frame * frame_shift

    midpoint_of_frame = frame_shift * frame + frame_shift // 2
    beginning_of_frame = midpoint_of_frame - opts.window_size // 2
    return beginning_of_frame


def get_num_frames(num_samples: int, opts: FrameExtractionOptions, flush: bool = True):
    frame_shift = opts.window_shift
    frame_length = opts.window_size

    if opts.snip_edges:
        if num_samples < frame_length:
            return 0
        return 1 + (num_samples - frame_length) // frame_shift

    num_frames = (num_samples + (frame_shift // 2)) // frame_shift
    if flush:
        return num_frames

    end_sample_of_last_frame = get_first_sample_of_frame(num_frames - 1, opts) + frame_length
    while num_frames > 0 and end_sample_of_last_frame > num_samples:
        num_frames -= 1
        end_sample_of_last_frame -= frame_shift
    return num_frames


# https://kaldi-asr.org/doc/structkaldi_1_1FeatureWindowFunction.html#ab1279040022f8d5cda682001f7167e5d
class FeatureWindowFunction:
    def __init__(self, opts: FrameExtractionOptions):
        self.window = _get_window(
            opts.window_type,
            opts.window_size,
            opts.blackman_coeff,
        )


def dither_(window: Tensor, dither_value: float):
    window += torch.randn(window.shape, dtype=window.dtype, device=window.device) * dither_value


def preemphasize_(window: Tensor, preemph_coeff: float):
    assert 0.0 <= preemph_coeff <= 1.0
    window[..., 1:] -= preemph_coeff * window[..., :-1]
    window[..., 0] -= preemph_coeff * window[..., 0]


def _log_energy(wave: Tensor):
    assert wave.ndim == 2
    num_channels, num_frames = wave.shape
    w1 = wave.view(num_channels, 1, num_frames)
    w2 = wave.view(num_channels, num_frames, 1)
    device, dtype = wave.device, wave.dtype
    epsilon = torch.tensor(torch.finfo(dtype).eps, device=device)
    return torch.bmm(w1, w2).squeeze(2).squeeze(1).clamp_(min=epsilon).log_()


def process_window_(
    opts: FrameExtractionOptions,
    window_function: FeatureWindowFunction,
    window: Tensor,
) -> Tuple[Tensor, float]:
    if window.ndim != 2:
        raise ValueError(f"Expected the input Tensor to be 2D. Found {window.shape}")

    if opts.dither:
        dither_(window, opts.dither)
        # print('AFTER DITHER:')
        # print(window)

    if opts.remove_dc_offset:
        window -= window.mean(dim=-1, keepdim=True)
        # print('DC OFFSET REMOVED:')
        # print(window)

    log_energy_pre_window = _log_energy(window)

    if opts.preemph_coeff != 0.0:
        preemphasize_(window, opts.preemph_coeff)
        # print('PREEMPH:')
        # print(window)

    window *= window_function.window
    return log_energy_pre_window


def extract_window(
    sample_offset: int,
    wave: Tensor,
    f: int,
    opts: FrameExtractionOptions,
    window_function: FeatureWindowFunction,
) -> Tuple[Tensor, float]:
    if wave.ndim != 2:
        raise ValueError(f"Expected the input Tensor to be 2D. Found {wave.shape}")

    # assume channels-first
    num_channels, num_frames_in = wave.shape

    frame_length = opts.window_size
    frame_length_padded = opts.padded_window_size

    num_samples = sample_offset + num_frames_in
    start_sample = get_first_sample_of_frame(f, opts)
    end_sample = start_sample + frame_length

    wave_start = start_sample - sample_offset
    wave_end = wave_start + frame_length

    # TODO: support batch dim
    window = torch.zeros(num_channels, frame_length_padded, dtype=wave.dtype, device=wave.device)
    if wave_start >= 0 and wave_end <= wave.size(-1):
        window[:, :frame_length] = wave[:, wave_start : wave_start + frame_length]
    else:
        for s in range(frame_length):
            s_in_wave = s + wave_start
            while s_in_wave < 0 or s_in_wave >= num_frames_in:
                if s_in_wave < 0:
                    s_in_wave = -s_in_wave - 1
                else:
                    s_in_wave = 2 * num_frames_in - 1 - s_in_wave
            window[:, s] = wave[:, s_in_wave]

    frame = window[:, :frame_length]

    # print("INPUT FRAME:")
    # print(frame.to(torch.int32))
    log_energy = process_window_(opts, window_function, frame)
    # print("LOG_ENERGY:", log_energy)
    # print("PROCESSED FRAME")
    # print(frame)
    return window, log_energy
