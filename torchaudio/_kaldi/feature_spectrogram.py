import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .feature_window import FrameExtractionOptions, _log_energy


# Based off of
# https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-spectrogram.h#L38
@dataclass
class SpectrogramOptions:
    frame_opts: FrameExtractionOptions
    energy_floor: Optional[float] = None
    raw_energy: bool = True
    return_raw_fft: bool = False


# https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-spectrogram.h#L67
class SpectrogramComputer:
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-spectrogram.cc#L27-L35
    def __init__(self, opts: SpectrogramOptions):
        self._opts = opts
        self._log_energy_floor = None

        if opts.energy_floor is not None:
            if opts.energy_floor <= 0.0:
                raise ValueError("The value of 'energy_floor' must be positive.")
            self._log_energy_floor = math.log(opts.energy_floor)

        # srfft_ optimization is omitted
        # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-spectrogram.cc#L32-L34

    @property
    def dim(self):
        ws = self._opts.frame_opts.padded_window_size
        if self._opts.return_raw_fft:
            return ws
        return ws // 2 + 1

    @property
    def frame_options(self):
        return self._opts.frame_opts

    @property
    def need_raw_log_energy(self):
        return self._opts.raw_energy

    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-spectrogram.cc#L47
    def compute(
        self,
        signal_raw_log_energy: Tensor,
        signal_frame: Tensor,
    ) -> Tensor:
        if signal_frame.ndim != 2:
            raise ValueError(f"Expected the input Tensor to be 2D. Found {signal_frame.shape}")
        num_channels, num_frames = signal_frame.shape
        if signal_raw_log_energy is not None and signal_raw_log_energy.numel() != num_channels:
            raise ValueError(
                "When signal_raw_log_energy is provided, "
                "it has to have the same number of elements as the input signal channels."
            )

        device, dtype = signal_frame.device, signal_frame.dtype
        epsilon = torch.tensor(torch.finfo(dtype).eps, device=device)
        # print("SIGNAL_FRAME:", signal_frame.shape)
        # print(signal_frame)
        # print("FEATURE_DIM:", self.dim)
        if not self._opts.raw_energy:
            signal_raw_log_energy = _log_energy(signal_frame)
            # print("RAW_ENERGY:", signal_raw_log_energy)

        # print("BEFORE RFFT: ", signal_frame)
        feature = torch.fft.rfft(signal_frame)  # , norm="forward")
        # print("AFTER RFFT:", feature.shape)
        # print(feature)

        if self._opts.return_raw_fft:
            return feature

        # power_spectrum = feature.abs().pow(2.0)
        # print("POWER_SPECTRUM:", power_spectrum)
        feature = feature.abs()
        feature.pow_(2.0).clamp_(min=epsilon).log_()

        if self._opts.energy_floor is not None:
            signal_raw_log_energy[signal_raw_log_energy < self._log_energy_floor] = self._log_energy_floor
        # print("LOG_ENERGY_FLOOR:", self._log_energy_floor)
        # print("RAW_ENERGY:", signal_raw_log_energy)
        feature[:, 0] = signal_raw_log_energy
        # print("FEATURES:", feature)

        return feature
