"""Simplified port of

https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-fbank.h

"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .feature_window import FrameExtractionOptions, _log_energy
from .mel_computations import MelBanksOptions, MelBanks


@dataclass
class FbankOptions:
    # Based off of
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-fbank.h#L41
    frame_opts: FrameExtractionOptions
    mel_opts: MelBanksOptions = MelBanksOptions(23)
    use_energy: bool = False
    energy_floor: float = 0.0
    raw_energy: bool = True
    htk_compat: bool = False
    use_log_fbank: bool = True
    use_power: bool = True


# https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-fbank.h#L86
# Main difference:
# - No caching of multiple MelBanks objects (Retain only one MelBanks instance) and move vtln_warp option to constructor
# - Omitting srfft_, which seems to be an optimization (and no direct equivalent in PyTorch)
class FbankComputer:
    # Based off of
    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-fbank.cc#L26
    def __init__(self, opts: FbankOptions, vtln_warp: Optional[float] = None):
        self._opts: FbankOptions = opts
        self._log_energy_floor: Optional[float] = None
        if opts.energy_floor > 0.0:
            self._log_energy_floor = math.log(opts.energy_floor)

        # srfft_ optimization is omitted
        # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-fbank.cc#L32-L33

        # Note:
        # In the original Kaldi, the main computation logic accepts local vtln_warp value,
        # which instantiates and caches new MelBanks instances.
        # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-fbank.cc#L35-L37
        #
        # The reasoning behind of such design seems to be for CLI.
        # Such design allows to use different vtln_warp values within one CLI invocations.
        # In our port, we do not need that.
        #
        self._mel_banks: MelBanks = MelBanks(self._opts.mel_opts, self._opts.frame_opts, vtln_warp)

    @property
    def dim(self) -> int:
        return self._opts.mel_opts.num_bins + 1 if self._opts.use_energy else 0

    @property
    def need_raw_log_energy(self) -> bool:
        return self._opts.use_energy and self._opts.raw_energy

    @property
    def frame_options(self):
        return self._opts.frame_opts

    # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/feat/feature-fbank.cc#L72
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

        if not self._opts.raw_energy:
            signal_raw_log_energy = _log_energy(signal_frame)

        feature = torch.fft.rfft(signal_frame)  # , norm="forward")
        # print('AFTER RFFT:', feature)

        feature = feature.abs()
        feature.pow_(2.0)

        # TODO: merge this with the above
        if not self._opts.use_power:
            feature.pow_(0.5)

        # print('AFTER USE POWER:', feature)

        # mel_offset = 1 if self._opts.use_energy else 0
        offset = 1 if self._opts.use_energy else 0
        # mel_energies = feature[..., mel_offset:mel_offset + self._opts.mel_opts.num_bins]
        # TODO: consider the use of `out` parameter
        num_feats = self._opts.mel_opts.num_bins + offset
        mel_energies = torch.empty([num_channels, num_feats], device=device, dtype=dtype)
        mel_energies[:, offset:] = self._mel_banks.compute(feature)
        # print('mel_energies.shape:', mel_energies.shape)
        # print(mel_energies)

        if self._opts.use_log_fbank:
            mel_energies.clamp_(min=epsilon)
            mel_energies.log_()

        if self._opts.use_energy:
            signal_raw_log_energy[signal_raw_log_energy < self._log_energy_floor] = self._log_energy_floor
            # print("LOG_ENERGY_FLOOR:", self._log_energy_floor)
            # print("RAW_ENERGY:", signal_raw_log_energy)
            mel_energies[:, 0] = signal_raw_log_energy
            # print("FEATURES:", feature)
        # print('feature:', mel_energies.shape)
        # print('feature:', mel_energies)
        return mel_energies
