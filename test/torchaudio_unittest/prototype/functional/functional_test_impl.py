import numpy as np
try:
    import pyroomacoustics as pra
except Exception:
    pass
import torch
import torchaudio.prototype.functional as F
from parameterized import parameterized
from scipy import signal
from torchaudio_unittest.common_utils import nested_params, skipIfNoModule, TestBaseMixin


class FunctionalTestImpl(TestBaseMixin):
    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
    )
    def test_convolve_numerics(self, leading_dims, lengths):
        """Check that convolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.convolve(x, y)

        num_signals = torch.tensor(leading_dims).prod() if leading_dims else 1
        x_reshaped = x.reshape((num_signals, L_x))
        y_reshaped = y.reshape((num_signals, L_y))
        expected = [
            signal.convolve(x_reshaped[i].detach().cpu().numpy(), y_reshaped[i].detach().cpu().numpy())
            for i in range(num_signals)
        ]
        expected = torch.tensor(np.array(expected))
        expected = expected.reshape(leading_dims + (-1,))

        self.assertEqual(expected, actual)

    @nested_params(
        [(10, 4), (4, 3, 1, 2), (2,), ()],
        [(100, 43), (21, 45)],
    )
    def test_fftconvolve_numerics(self, leading_dims, lengths):
        """Check that fftconvolve returns values identical to those that SciPy produces."""
        L_x, L_y = lengths

        x = torch.rand(*(leading_dims + (L_x,)), dtype=self.dtype, device=self.device)
        y = torch.rand(*(leading_dims + (L_y,)), dtype=self.dtype, device=self.device)

        actual = F.fftconvolve(x, y)

        expected = signal.fftconvolve(x.detach().cpu().numpy(), y.detach().cpu().numpy(), axes=-1)
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
    @parameterized.expand([(2, 1), (3, 4)])
    def test_simulate_rir_ism_single_band(self, D, channel):
        """Test simulate_rir_ism function in the case where absorption coefficients are identical for all walls."""
        room_dim = torch.rand(D, dtype=self.dtype, device=self.device) + 5
        mic_array = torch.rand(channel, D, dtype=self.dtype, device=self.device) + 1
        source = torch.rand(D, dtype=self.dtype, device=self.device) + 4
        max_order = 3
        e_absorption = 0.5
        room = pra.ShoeBox(
            room_dim.detach().numpy(),
            fs=16000,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            ray_tracing=False,
            air_absorption=False,
        )
        mic_locs = np.asarray([mic_array[i].tolist() for i in range(channel)]).swapaxes(0, 1)
        room.add_microphone_array(mic_locs)
        room.add_source(source.tolist())
        room.compute_rir()
        max_len = max([room.rir[i][0].shape[0] for i in range(channel)])
        expected = torch.zeros(channel, max_len, dtype=self.dtype, device=self.device)
        for i in range(channel):
            expected[i, 0 : room.rir[i][0].shape[0]] = torch.from_numpy(room.rir[i][0])
        actual = F.simulate_rir_ism(room_dim, source, mic_array, max_order, e_absorption)
        self.assertEqual(expected, actual, atol=1e-3, rtol=2)

    @skipIfNoModule("pyroomacoustics")
    @parameterized.expand([(2, 1), (3, 4)])
    def test_simulate_rir_ism_multi_band(self, D, channel):
        """Test simulate_rir_ism in the case where absorption coefficients are different for all walls."""
        room_dim = torch.rand(D, dtype=self.dtype, device=self.device) + 5
        mic_array = torch.rand(channel, D, dtype=self.dtype, device=self.device) + 1
        source = torch.rand(D, dtype=self.dtype, device=self.device) + 4
        max_order = 3
        if D == 2:
            e_absorption = torch.rand(7, 4, dtype=self.dtype, device=self.device)
            walls = ["west", "east", "south", "north"]
        else:
            e_absorption = torch.rand(7, 6, dtype=self.dtype, device=self.device)
            walls = ["west", "east", "south", "north", "floor", "ceiling"]
        room = pra.ShoeBox(
            room_dim.detach().numpy(),
            fs=16000,
            materials={
                walls[i]: pra.Material(
                    {
                        "coeffs": e_absorption[:, i]
                        .reshape(
                            -1,
                        )
                        .detach()
                        .numpy(),
                        "center_freqs": [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
                    }
                )
                for i in range(len(walls))
            },
            max_order=max_order,
            ray_tracing=False,
            air_absorption=False,
        )
        mic_locs = np.asarray([mic_array[i].tolist() for i in range(channel)]).swapaxes(0, 1)
        room.add_microphone_array(mic_locs)
        room.add_source(source.tolist())
        room.compute_rir()
        max_len = max([room.rir[i][0].shape[0] for i in range(channel)])
        expected = torch.zeros(channel, max_len, dtype=self.dtype, device=self.device)
        for i in range(channel):
            expected[i, 0 : room.rir[i][0].shape[0]] = torch.from_numpy(room.rir[i][0])
        actual = F.simulate_rir_ism(room_dim, source, mic_array, max_order, e_absorption)
        self.assertEqual(expected, actual, atol=1e-3, rtol=2)
