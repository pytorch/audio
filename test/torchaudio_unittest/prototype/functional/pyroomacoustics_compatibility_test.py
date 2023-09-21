import math

import numpy as np
import torch
import torchaudio.prototype.functional as F

from parameterized import parameterized
from torchaudio._internal import module_utils as _mod_utils
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoModule, skipIfNoRIR

if _mod_utils.is_module_available("pyroomacoustics"):
    import pyroomacoustics as pra


def _pra_ray_tracing(
    room_dim,
    absorption,
    scattering,
    num_bands,
    mic_array,
    source,
    num_rays,
    energy_thres,
    time_thres,
    hist_bin_size,
    mic_radius,
    sound_speed,
):
    walls = ["west", "east", "south", "north", "floor", "ceiling"]
    absorption = absorption.T.tolist()
    scattering = scattering.T.tolist()
    freqs = 125 * 2 ** np.arange(num_bands)

    room = pra.ShoeBox(
        room_dim.tolist(),
        ray_tracing=True,
        materials={
            wall: pra.Material(
                energy_absorption={"coeffs": absorp, "center_freqs": freqs},
                scattering={"coeffs": scat, "center_freqs": freqs},
            )
            for wall, absorp, scat in zip(walls, absorption, scattering)
        },
        air_absorption=False,
        max_order=0,  # Make sure PRA doesn't use the hybrid method (we just want ray tracing)
    )
    room.add_microphone_array(mic_array.T.tolist())
    room.add_source(source.tolist())
    room.set_ray_tracing(
        n_rays=num_rays,
        energy_thres=energy_thres,
        time_thres=time_thres,
        hist_bin_size=hist_bin_size,
        receiver_radius=mic_radius,
    )
    room.set_sound_speed(sound_speed)
    room.compute_rir()
    hist_pra = np.array(room.rt_histograms, dtype=np.float32)[:, 0, 0]

    # PRA continues the simulation beyond time threshold, but torchaudio does not.
    num_bins = math.ceil(time_thres / hist_bin_size)
    return hist_pra[:, :, :num_bins]


@skipIfNoModule("pyroomacoustics")
@skipIfNoRIR
class CompatibilityTest(PytorchTestCase):

    # pyroomacoustics uses float for internal implementations.
    dtype = torch.float32
    device = torch.device("cpu")

    @parameterized.expand([(1,), (4,)])
    def test_simulate_rir_ism_single_band(self, channel):
        """Test simulate_rir_ism function in the case where absorption coefficients are identical for all walls."""
        room_dim = torch.rand(3, dtype=self.dtype, device=self.device) + 5
        mic_array = torch.rand(channel, 3, dtype=self.dtype, device=self.device) + 1
        source = torch.rand(3, dtype=self.dtype, device=self.device) + 4
        max_order = 3
        # absorption is set as a float value indicating absorption coefficients are the same for every wall.
        absorption = 0.5
        # compute rir signal by torchaudio implementation
        actual = F.simulate_rir_ism(room_dim, source, mic_array, max_order, absorption)
        # compute rir signal by pyroomacoustics
        room = pra.ShoeBox(
            room_dim.detach().numpy(),
            fs=16000,
            materials=pra.Material(absorption),
            max_order=max_order,
            ray_tracing=False,
            air_absorption=False,
        )
        # mic_locs is a numpy array of dimension `(3, channel)`.
        mic_locs = mic_array.transpose(0, 1).double().detach().numpy()
        room.add_microphone_array(mic_locs)
        room.add_source(source.tolist())
        room.compute_rir()
        max_len = max(room.rir[i][0].shape[0] for i in range(channel))
        expected = torch.zeros(channel, max_len, dtype=self.dtype, device=self.device)
        for i in range(channel):
            expected[i, 0 : room.rir[i][0].shape[0]] = torch.from_numpy(room.rir[i][0])

        self.assertEqual(expected, actual, atol=1e-3, rtol=1e-3)

    @parameterized.expand([(1,), (4,)])
    def test_simulate_rir_ism_multi_band(self, channel):
        """Test simulate_rir_ism in the case where absorption coefficients are different for all walls."""
        room_dim = torch.rand(3, dtype=self.dtype, device=self.device) + 5
        mic_array = torch.rand(channel, 3, dtype=self.dtype, device=self.device) + 1
        source = torch.rand(3, dtype=self.dtype, device=self.device) + 4
        max_order = 3
        # absorption is set as a Tensor with dimensions `(7, 6)` indicating there are
        # 6 walls and each wall has 7 absorption coefficients corresponds to 7 octave bands, respectively.
        absorption = torch.rand(7, 6, dtype=self.dtype, device=self.device)
        walls = ["west", "east", "south", "north", "floor", "ceiling"]
        room = pra.ShoeBox(
            room_dim.detach().numpy(),
            fs=16000,
            materials={
                walls[i]: pra.Material(
                    {
                        "coeffs": absorption[:, i]
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
        # mic_locs is a numpy array of dimension `(D, channel)`.
        mic_locs = mic_array.transpose(0, 1).double().detach().numpy()
        room.add_microphone_array(mic_locs)
        room.add_source(source.tolist())
        room.compute_rir()
        max_len = max(room.rir[i][0].shape[0] for i in range(channel))
        expected = torch.zeros(channel, max_len, dtype=self.dtype, device=self.device)
        for i in range(channel):
            expected[i, 0 : room.rir[i][0].shape[0]] = torch.from_numpy(room.rir[i][0])
        actual = F.simulate_rir_ism(room_dim, source, mic_array, max_order, absorption)
        self.assertEqual(expected, actual, atol=1e-3, rtol=1e-3)

    @parameterized.expand(
        [
            ([20, 25, 30], [1, 10, 5], [[8, 8, 22]], 130),
        ]
    )
    def test_ray_tracing_same_results_as_pyroomacoustics(self, room, source, mic_array, num_rays):
        num_bands = 6
        energy_thres = 1e-7
        time_thres = 10.0
        hist_bin_size = 0.004
        mic_radius = 0.5
        sound_speed = 343.0

        absorption = torch.full((num_bands, 6), 0.1, dtype=self.dtype)
        scattering = torch.full((num_bands, 6), 0.4, dtype=self.dtype)
        room = torch.tensor(room, dtype=self.dtype)
        source = torch.tensor(source, dtype=self.dtype)
        mic_array = torch.tensor(mic_array, dtype=self.dtype)

        hist_pra = _pra_ray_tracing(
            room,
            absorption,
            scattering,
            num_bands,
            mic_array,
            source,
            num_rays,
            energy_thres,
            time_thres,
            hist_bin_size,
            mic_radius,
            sound_speed,
        )

        hist = F.ray_tracing(
            room=room,
            source=source,
            mic_array=mic_array,
            num_rays=num_rays,
            absorption=absorption,
            scattering=scattering,
            sound_speed=sound_speed,
            mic_radius=mic_radius,
            energy_thres=energy_thres,
            time_thres=time_thres,
            hist_bin_size=hist_bin_size,
        )

        self.assertEqual(hist, hist_pra, atol=0.001, rtol=0.001)
