import unittest

import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin, torch_script


class TorchScriptConsistencyTestImpl(TestBaseMixin):
    def _assert_consistency(self, func, inputs, shape_only=False):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(device=self.device, dtype=self.dtype)
            inputs_.append(i)
        ts_func = torch_script(func)

        torch.random.manual_seed(40)
        output = func(*inputs_)

        torch.random.manual_seed(40)
        ts_output = ts_func(*inputs_)

        if shape_only:
            ts_output = ts_output.shape
            output = output.shape
        self.assertEqual(ts_output, output)

    @nested_params(
        [F.convolve, F.fftconvolve],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (2, 3, 2)
        L_x, L_y = 32, 55
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        self._assert_consistency(fn, (x, y, mode))

    def test_add_noise(self):
        leading_dims = (2, 3)
        L = 31

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        self._assert_consistency(F.add_noise, (waveform, noise, lengths, snr))

    def test_barkscale_fbanks(self):
        if self.device != torch.device("cpu"):
            raise unittest.SkipTest("No need to perform test on device other than CPU")

        n_stft = 100
        f_min = 0.0
        f_max = 20.0
        n_barks = 10
        sample_rate = 16000
        self._assert_consistency(F.barkscale_fbanks, (n_stft, f_min, f_max, n_barks, sample_rate, "traunmuller"))

    @parameterized.expand(
        [
            ([20, 25], [2, 2], [[8, 8], [7, 6]], 1_000),  # 2D with 2 mics
            ([20, 25, 30], [1, 10, 5], [[8, 8, 22]], 500),  # 3D with 1 mic
        ]
    )
    def test_ray_tracing(self, room_dim, source, mic_array, num_rays):
        num_walls = 4 if len(room_dim) == 2 else 6
        num_bands = 3

        e_absorption = torch.rand(num_bands, num_walls, dtype=torch.float32)
        scattering = torch.rand(num_bands, num_walls, dtype=torch.float32)

        energy_thres = 1e-7
        time_thres = 10
        hist_bin_size = 0.004
        mic_radius = 0.5
        sound_speed = 343

        room_dim = torch.tensor(room_dim, dtype=self.dtype)
        source = torch.tensor(source, dtype=self.dtype)
        mic_array = torch.tensor(mic_array, dtype=self.dtype)

        self._assert_consistency(
            F.ray_tracing,
            (
                room_dim,
                source,
                mic_array,
                num_rays,
                e_absorption,
                scattering,
                mic_radius,
                sound_speed,
                energy_thres,
                time_thres,
                hist_bin_size,
            ),
        )

    def test_oscillator_bank(self):
        num_frames, num_pitches, sample_rate = 8000, 8, 8000
        freq = torch.rand((num_frames, num_pitches), dtype=self.dtype, device=self.device)
        amps = torch.ones_like(freq)

        self._assert_consistency(F.oscillator_bank, (freq, amps, sample_rate, "sum"))
