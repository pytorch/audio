import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .feature_window import FrameExtractionOptions, _log_energy
from .mel_computations import MelBanksOptions, MelBanks, compute_lifter_coeffs


@dataclass
class MfccOptions:
    frame_opts: FrameExtractionOptions
    mel_opts: MelBanksOptions = MelBanksOptions(23)
    num_ceps: int = 13
    use_energy: bool = True
    energy_floor: Optional[float] = None
    raw_energy: bool = True
    cepstral_lifter: Optional[float] = 22.0


def _compute_dct_matrix(num_rows, num_cols):
    mat = torch.empty(num_rows, num_cols)
    normalizer = math.sqrt(1.0 / num_cols)
    for i in range(num_cols):
        mat[0, i] = normalizer
    normalizer = math.sqrt(2.0 / num_cols)
    for i in range(1, num_rows):
        for j in range(num_cols):
            mat[i, j] = normalizer * math.cos(math.pi / num_cols * (j + 0.5) * i)
    return mat


class MfccComputer:
    def __init__(self, opts: MfccOptions, vtln_warp: Optional[float] = None):
        self._opts = opts

        num_bins = opts.mel_opts.num_bins
        if opts.num_ceps > num_bins:
            raise ValueError(
                "'num_ceps' cannot be larger than the number of mel bins. "
                f"num_ceps: {opts.num_ceps}, mel_opts.num_bins: {num_bins}."
            )

        self._dct_matrix = _compute_dct_matrix(opts.num_ceps, num_bins).T
        # print("dct_matrix:", self._dct_matrix.T.shape)
        # print(self._dct_matrix.T)

        self._lifter_coeffs: Optional[float] = None
        if opts.cepstral_lifter is not None:
            self._lifter_coeffs = compute_lifter_coeffs(opts.cepstral_lifter, opts.num_ceps)

        self._log_energy_floor: Optional[float] = None
        if opts.energy_floor is not None:
            self._log_energy_floor = math.log(opts.energy_floor)

        self._mel_banks: MelBanks = MelBanks(opts.mel_opts, opts.frame_opts, vtln_warp)

    @property
    def frame_options(self):
        return self._opts.frame_opts

    @property
    def dim(self):
        return self._opts.num_ceps

    @property
    def need_raw_log_energy(self):
        return self._opts.use_energy and self._opts.raw_energy

    def compute(self, signal_raw_log_energy: Tensor, signal_frame: Tensor) -> Tensor:
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

        if self._opts.use_energy and not self._opts.raw_energy:
            signal_raw_log_energy = _log_energy(signal_frame)

        feature = torch.fft.rfft(signal_frame)

        feature = feature.abs()
        feature.pow_(2.0)

        mel_energies = self._mel_banks.compute(feature)
        mel_energies.clamp_(min=epsilon)
        mel_energies.log_()

        # print("mel_energies:", mel_energies)
        feature = torch.matmul(mel_energies, self._dct_matrix)
        # print("feature:", feature)

        if self._opts.cepstral_lifter is not None:
            feature *= self._lifter_coeffs
            # print("feature (lifter):", feature)

        if self._opts.use_energy:
            if self._opts.energy_floor is not None:
                signal_raw_log_energy[signal_raw_log_energy < self._log_energy_floor] = self._log_energy_floor
            feature[:, 0] = signal_raw_log_energy

        return feature
