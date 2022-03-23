import torch
from torch import Tensor

from .feature_window import FeatureWindowFunction, get_num_frames, extract_window


class OfflineFeatureTpl:
    def __init__(self, computer, subtract_mean: bool = False):
        self._computer = computer
        self._subtract_mean = subtract_mean
        self._feature_window_function = FeatureWindowFunction(computer.frame_options)

    def compute(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"Expected the input Tensor to be 2D. Found {waveform.shape}")

        # assume channels-first
        num_channels, num_frames_in = waveform.shape

        dtype, device = waveform.dtype, waveform.device

        rows_out = get_num_frames(num_frames_in, self._computer.frame_options)
        cols_out = self._computer.dim
        print("rows_out, cols_out:", rows_out, cols_out)

        # assume output to be [C, T, Feat dim].
        # TODO: revise this decision
        output = torch.empty(num_channels, rows_out, cols_out, dtype=dtype, device=device)

        if rows_out == 0:
            return output

        use_raw_log_energy = self._computer.need_raw_log_energy

        features = []
        for r in range(rows_out):

            windowed, log_energy_pre_window = extract_window(
                0, waveform, r, self._computer.frame_options, self._feature_window_function
            )
            # print("windowed.shape:", windowed.shape)

            raw_log_energy = log_energy_pre_window if use_raw_log_energy else None
            feature = self._computer.compute(raw_log_energy, windowed)
            features.append(feature)
            # if r == 3:
            #     raise RuntimeError(f"STOP AFTER SOME ITERATIONS. {feature}")
            # else:
            #     print(feature)

        # TODO: revise the final shape. C,T,F or C,F,T
        stack_dim = 1
        features = torch.stack(features, dim=stack_dim)
        if self._subtract_mean:
            # Note:
            # The code block makes the descrepancy with Kaldi larger
            # Such as atol=1e-3 (from 1e-6)
            #
            # The routine here causes the descrepancy.
            # https://github.com/kaldi-asr/kaldi/blob/dd107fd594ac58af962031c1689abfdc10f84452/src/featbin/compute-spectrogram-feats.cc#L126-L132
            features -= features.mean(dim=stack_dim, keepdim=True)
        return features
