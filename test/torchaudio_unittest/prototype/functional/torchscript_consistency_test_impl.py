import unittest

import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from torchaudio_unittest.common_utils import skipIfNoRIR, TestBaseMixin, torch_script


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

    def test_barkscale_fbanks(self):
        if self.device != torch.device("cpu"):
            raise unittest.SkipTest("No need to perform test on device other than CPU")

        n_stft = 100
        f_min = 0.0
        f_max = 20.0
        n_barks = 10
        sample_rate = 16000
        self._assert_consistency(F.barkscale_fbanks, (n_stft, f_min, f_max, n_barks, sample_rate, "traunmuller"))

    def test_oscillator_bank(self):
        num_frames, num_pitches, sample_rate = 8000, 8, 8000
        freq = torch.rand((num_frames, num_pitches), dtype=self.dtype, device=self.device)
        amps = torch.ones_like(freq)

        self._assert_consistency(F.oscillator_bank, (freq, amps, sample_rate, "sum", torch.float64))

    def test_extend_pitch(self):
        num_frames = 5
        input = torch.ones((num_frames, 1), device=self.device, dtype=self.dtype)

        num_pitches = 7
        pattern = [i + 1.0 for i in range(num_pitches)]

        self._assert_consistency(F.extend_pitch, (input, num_pitches))
        self._assert_consistency(F.extend_pitch, (input, pattern))
        self._assert_consistency(F.extend_pitch, (input, torch.tensor(pattern)))

    def test_sinc_ir(self):
        cutoff = torch.tensor([0, 0.5, 1.0], device=self.device, dtype=self.dtype)
        self._assert_consistency(F.sinc_impulse_response, (cutoff, 513, False))
        self._assert_consistency(F.sinc_impulse_response, (cutoff, 513, True))

    def test_freq_ir(self):
        mags = torch.tensor([0, 0.5, 1.0], device=self.device, dtype=self.dtype)
        self._assert_consistency(F.frequency_impulse_response, (mags,))


class TorchScriptConsistencyCPUOnlyTestImpl(TestBaseMixin):
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

    @skipIfNoRIR
    @parameterized.expand([(1,), (4,)])
    def test_simulate_rir_ism_single_band(self, channel):
        room_dim = torch.rand(3, dtype=self.dtype, device=self.device) + 5
        mic_array = torch.rand(channel, 3, dtype=self.dtype, device=self.device) + 1
        source = torch.rand(3, dtype=self.dtype, device=self.device) + 4
        max_order = 3
        absorption = 0.5
        center_frequency = torch.tensor([125, 250, 500, 1000, 2000, 4000, 8000], dtype=self.dtype, device=self.device)
        self._assert_consistency(
            F.simulate_rir_ism,
            (room_dim, source, mic_array, max_order, absorption, None, 81, center_frequency, 343.0, 16000.0),
        )

    @skipIfNoRIR
    @parameterized.expand([(1,), (4,)])
    def test_simulate_rir_ism_multi_band(self, channel):
        room_dim = torch.rand(3, dtype=self.dtype, device=self.device) + 5
        mic_array = torch.rand(channel, 3, dtype=self.dtype, device=self.device) + 1
        source = torch.rand(3, dtype=self.dtype, device=self.device) + 4
        max_order = 3
        absorption = torch.rand(7, 6, dtype=self.dtype, device=self.device)
        center_frequency = torch.tensor([125, 250, 500, 1000, 2000, 4000, 8000], dtype=self.dtype, device=self.device)
        self._assert_consistency(
            F.simulate_rir_ism,
            (room_dim, source, mic_array, max_order, absorption, None, 81, center_frequency, 343.0, 16000.0),
        )

    @parameterized.expand(
        [
            ([20, 25, 30], [1, 10, 5], [[8, 8, 22]], 500),  # 3D with 1 mic
        ]
    )
    def test_ray_tracing(self, room_dim, source, mic_array, num_rays):
        num_walls = 4 if len(room_dim) == 2 else 6
        num_bands = 3

        absorption = torch.rand(num_bands, num_walls, dtype=torch.float32)
        scattering = torch.rand(num_bands, num_walls, dtype=torch.float32)

        energy_thres = 1e-7
        time_thres = 10.0
        hist_bin_size = 0.004
        mic_radius = 0.5
        sound_speed = 343.0

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
                absorption,
                scattering,
                mic_radius,
                sound_speed,
                energy_thres,
                time_thres,
                hist_bin_size,
            ),
        )
