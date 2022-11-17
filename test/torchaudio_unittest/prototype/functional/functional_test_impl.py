import numpy as np
import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from scipy import signal
from torchaudio._internal import module_utils as _mod_utils
from torchaudio_unittest.common_utils import nested_params, skipIfNoModule, TestBaseMixin

if _mod_utils.is_module_available("pyroomacoustics"):
    import pyroomacoustics as pra


class FunctionalTestImpl(TestBaseMixin):
    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
        ["full", "valid", "same"],
    )
    def test_convolve_numerics(self, leading_dims, lengths, mode):
        """Check that convolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.convolve(x, y, mode=mode)

        num_signals = torch.tensor(leading_dims).prod() if leading_dims else 1
        x_reshaped = x.reshape((num_signals, L_x))
        y_reshaped = y.reshape((num_signals, L_y))
        expected = [
            signal.convolve(x_reshaped[i].detach().cpu().numpy(), y_reshaped[i].detach().cpu().numpy(), mode=mode)
            for i in range(num_signals)
        ]
        expected = torch.tensor(np.array(expected))
        expected = expected.reshape(leading_dims + (-1,))

        self.assertEqual(expected, actual)

    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
        ["full", "valid", "same"],
    )
    def test_fftconvolve_numerics(self, leading_dims, lengths, mode):
        """Check that fftconvolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.fftconvolve(x, y, mode=mode)

        expected = signal.fftconvolve(x.detach().cpu().numpy(), y.detach().cpu().numpy(), axes=-1, mode=mode)
        expected = torch.tensor(expected)

        self.assertEqual(expected, actual)

    @nested_params(
        [F.convolve, F.fftconvolve],
        [(4, 3, 1, 2), (1,)],
        [(10, 4), (2, 2, 2)],
    )
    def test_convolve_input_leading_dim_check(self, fn, x_shape, y_shape):
        """Check that convolve properly rejects inputs with different leading dimensions."""
        x = torch.rand(*x_shape, dtype=self.dtype, device=self.device)
        y = torch.rand(*y_shape, dtype=self.dtype, device=self.device)
        with self.assertRaisesRegex(ValueError, "Leading dimensions"):
            fn(x, y)

    def test_add_noise_broadcast(self):
        """Check that add_noise produces correct outputs when broadcasting input dimensions."""
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(5, 1, 1, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(5, 1, 3, dtype=self.dtype, device=self.device)
        snr = torch.rand(1, 1, 1, dtype=self.dtype, device=self.device) * 10
        actual = F.add_noise(waveform, noise, lengths, snr)

        noise_expanded = noise.expand(*leading_dims, L)
        snr_expanded = snr.expand(*leading_dims)
        lengths_expanded = lengths.expand(*leading_dims)
        expected = F.add_noise(waveform, noise_expanded, lengths_expanded, snr_expanded)

        self.assertEqual(expected, actual)

    @parameterized.expand(
        [((5, 2, 3), (2, 1, 1), (5, 2), (5, 2, 3)), ((2, 1), (5,), (5,), (5,)), ((3,), (5, 2, 3), (2, 1, 1), (5, 2))]
    )
    def test_add_noise_leading_dim_check(self, waveform_dims, noise_dims, lengths_dims, snr_dims):
        """Check that add_noise properly rejects inputs with different leading dimension lengths."""
        L = 51

        waveform = torch.rand(*waveform_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*noise_dims, L, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*lengths_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*snr_dims, dtype=self.dtype, device=self.device) * 10

        with self.assertRaisesRegex(ValueError, "Input leading dimensions"):
            F.add_noise(waveform, noise, lengths, snr)

    def test_add_noise_length_check(self):
        """Check that add_noise properly rejects inputs that have inconsistent length dimensions."""
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device)
        noise = torch.rand(*leading_dims, 50, dtype=self.dtype, device=self.device)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device) * 10

        with self.assertRaisesRegex(ValueError, "Length dimensions"):
            F.add_noise(waveform, noise, lengths, snr)

    @skipIfNoModule("pyroomacoustics")
    @parameterized.expand(
        [
            ([20, 25], [2, 2], [[8, 8], [7, 6]], 10_000),  # 2D with 2 mics
            ([20, 25, 30], [1, 10, 5], [8, 8, 22], 1_000),  # 3D with 1 mic
        ]
    )
    def test_ray_tracing_same_results_as_pyroomacoustics(self, room_dim, source, mic_array, num_rays):
        e_absorption = 0.2
        scattering = 0.2
        energy_thres = 1e-7
        time_thres = 10
        hist_bin_size = 0.004
        mic_radius = 0.5
        sound_speed = 343

        room_dim = torch.tensor(room_dim, dtype=self.dtype)
        source = torch.tensor(source, dtype=self.dtype)
        mic_array = torch.tensor(mic_array, dtype=self.dtype)

        room = pra.ShoeBox(
            room_dim.tolist(),
            ray_tracing=True,
            materials=pra.Material(energy_absorption=e_absorption, scattering=scattering),
            air_absorption=False,
        )
        room.add_microphone(mic_array.tolist())
        room.add_source(source.tolist())
        room.set_ray_tracing(
            n_rays=num_rays,
            energy_thres=energy_thres,
            time_thres=time_thres,
            hist_bin_size=hist_bin_size,
            receiver_radius=mic_radius,
        )
        room.set_sound_speed(sound_speed)
        room.is_hybrid_sim = False

        room.compute_rir()
        hist_pra = torch.tensor(room.rt_histograms)[:, 0, 0, 0, :]  # shape = (num_mics, hist_size)

        hist = F.ray_tracing(
            room=room_dim,
            source=source,
            mic_array=mic_array,
            num_rays=num_rays,
            e_absorption=e_absorption,
            scattering=scattering,
            sound_speed=sound_speed,
            mic_radius=mic_radius,
            energy_thres=energy_thres,
            time_thres=time_thres,
            hist_bin_size=hist_bin_size,
        )

        if hist.dim() == 2:
            assert hist.shape[0] == mic_array.shape[0]  # one histogram per mic
        else:
            assert hist.dim() == 1

        # Most histogram entries are very very close, but very few of them can
        # be slightly off. We thus assert that less than 1% entry are off by
        # more than atol
        percent = 1 / 100
        atol = 1e-4
        diff = abs(hist_pra - hist.to(torch.float32))
        assert ((diff > atol).float().mean(axis=1) < percent).all()
