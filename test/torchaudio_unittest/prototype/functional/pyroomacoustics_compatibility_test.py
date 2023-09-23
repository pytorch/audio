import torch
import torchaudio.prototype.functional as F

from parameterized import parameterized
from torchaudio._internal import module_utils as _mod_utils
from torchaudio_unittest.common_utils import PytorchTestCase, skipIfNoModule, skipIfNoRIR

if _mod_utils.is_module_available("pyroomacoustics"):
    import pyroomacoustics as pra


@skipIfNoModule("pyroomacoustics")
@skipIfNoRIR
class CompatibilityTest(PytorchTestCase):

    dtype = torch.float64
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
