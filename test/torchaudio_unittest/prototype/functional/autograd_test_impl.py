import torch
import torchaudio.prototype.functional as F
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import nested_params, TestBaseMixin


class AutogradTestImpl(TestBaseMixin):
    @nested_params(
        [F.convolve, F.fftconvolve],
        ["full", "valid", "same"],
    )
    def test_convolve(self, fn, mode):
        leading_dims = (4, 3, 2)
        L_x, L_y = 23, 40
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device, requires_grad=True)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device, requires_grad=True)
        self.assertTrue(gradcheck(fn, (x, y, mode)))
        self.assertTrue(gradgradcheck(fn, (x, y, mode)))

    def test_add_noise(self):
        leading_dims = (5, 2, 3)
        L = 51

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        self.assertTrue(gradcheck(F.add_noise, (waveform, noise, lengths, snr)))
        self.assertTrue(gradgradcheck(F.add_noise, (waveform, noise, lengths, snr)))


class AutogradTestRayTracingImpl(TestBaseMixin):
    # @parameterized.expand([(2, 1), (3, 4)])
    def test_simulate_rir_ism(self):

        room_dim = [20, 25]
        source = [2, 2]
        mic_array = [8, 8]
        num_rays = 1_000

        e_absorption = 0.2
        scattering = 0.2

        room_dim = torch.tensor(room_dim, dtype=self.dtype, requires_grad=True)
        source = torch.tensor(source, dtype=self.dtype, requires_grad=True)
        mic_array = torch.tensor(mic_array, dtype=self.dtype, requires_grad=True)

        # TODO: make this work
        # self.assertTrue(
        #     gradcheck(
        #         F.ray_tracing,
        #         (
        #             room_dim,
        #             source,
        #             mic_array,
        #             num_rays,
        #             e_absorption,
        #             scattering,
        #         ),
        #         atol=1e-3,
        #         rtol=1,
        #     )
        # )

        # self.assertTrue(
        #     gradgradcheck(
        #         F.ray_tracing,
        #         (
        #             room_dim,
        #             source,
        #             mic_array,
        #             num_rays,
        #             e_absorption,
        #             scattering,
        #         ),
        #         atol=1e-3,
        #         rtol=1,
        #     )
        # )
