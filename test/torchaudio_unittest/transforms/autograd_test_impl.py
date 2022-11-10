import unittest
from typing import List

import torch
import torchaudio.transforms as T
from parameterized import parameterized
from torch.autograd import gradcheck, gradgradcheck
from torchaudio_unittest.common_utils import get_spectrogram, get_whitenoise, nested_params, rnnt_utils, TestBaseMixin


class _DeterministicWrapper(torch.nn.Module):
    """Helper transform wrapper to make the given transform deterministic"""

    def __init__(self, transform, seed=0):
        super().__init__()
        self.seed = seed
        self.transform = transform

    def forward(self, input: torch.Tensor):
        torch.random.manual_seed(self.seed)
        return self.transform(input)


class AutogradTestMixin(TestBaseMixin):
    def assert_grad(
        self,
        transform: torch.nn.Module,
        inputs: List[torch.Tensor],
        *,
        nondet_tol: float = 0.0,
    ):
        transform = transform.to(dtype=torch.float64, device=self.device)

        # gradcheck and gradgradcheck only pass if the input tensors are of dtype `torch.double` or
        # `torch.cdouble`, when the default eps and tolerance values are used.
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(dtype=torch.cdouble if i.is_complex() else torch.double, device=self.device)
                i.requires_grad = True
            inputs_.append(i)
        assert gradcheck(transform, inputs_)
        assert gradgradcheck(transform, inputs_, nondet_tol=nondet_tol)

    @parameterized.expand(
        [
            ({"pad": 0, "normalized": False, "power": None, "return_complex": True},),
            ({"pad": 3, "normalized": False, "power": None, "return_complex": True},),
            ({"pad": 0, "normalized": True, "power": None, "return_complex": True},),
            ({"pad": 3, "normalized": True, "power": None, "return_complex": True},),
            ({"pad": 0, "normalized": False, "power": None},),
            ({"pad": 3, "normalized": False, "power": None},),
            ({"pad": 0, "normalized": True, "power": None},),
            ({"pad": 3, "normalized": True, "power": None},),
            ({"pad": 0, "normalized": False, "power": 1.0},),
            ({"pad": 3, "normalized": False, "power": 1.0},),
            ({"pad": 0, "normalized": True, "power": 1.0},),
            ({"pad": 3, "normalized": True, "power": 1.0},),
            ({"pad": 0, "normalized": False, "power": 2.0},),
            ({"pad": 3, "normalized": False, "power": 2.0},),
            ({"pad": 0, "normalized": True, "power": 2.0},),
            ({"pad": 3, "normalized": True, "power": 2.0},),
        ]
    )
    def test_spectrogram(self, kwargs):
        # replication_pad1d_backward_cuda is not deteministic and
        # gives very small (~2.7756e-17) difference.
        #
        # See https://github.com/pytorch/pytorch/issues/54093
        transform = T.Spectrogram(**kwargs)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    def test_inverse_spectrogram(self):
        # create a realistic input:
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        length = waveform.shape[-1]
        spectrogram = get_spectrogram(waveform, n_fft=400)
        # test
        inv_transform = T.InverseSpectrogram(n_fft=400)
        self.assert_grad(inv_transform, [spectrogram, length])

    def test_melspectrogram(self):
        # replication_pad1d_backward_cuda is not deteministic and
        # gives very small (~2.7756e-17) difference.
        #
        # See https://github.com/pytorch/pytorch/issues/54093
        sample_rate = 8000
        transform = T.MelSpectrogram(sample_rate=sample_rate)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)



    @nested_params(
        [0, 0.99],
        [False, True],
    )
    def test_griffinlim(self, momentum, rand_init):
        n_fft = 80
        power = 1
        n_iter = 2

        spec = get_spectrogram(get_whitenoise(sample_rate=8000, duration=0.01, n_channels=2), n_fft=n_fft, power=power)
        transform = _DeterministicWrapper(
            T.GriffinLim(n_fft=n_fft, n_iter=n_iter, momentum=momentum, rand_init=rand_init, power=power)
        )
        self.assert_grad(transform, [spec])

    @parameterized.expand([(False,), (True,)])
    def test_mfcc(self, log_mels):
        sample_rate = 8000
        transform = T.MFCC(sample_rate=sample_rate, log_mels=log_mels)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    @parameterized.expand([(False,), (True,)])
    def test_lfcc(self, log_lf):
        sample_rate = 8000
        transform = T.LFCC(sample_rate=sample_rate, log_lf=log_lf)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    def test_compute_deltas(self):
        transform = T.ComputeDeltas()
        spec = torch.rand(10, 20)
        self.assert_grad(transform, [spec])

    @parameterized.expand([(8000, 8000), (8000, 4000), (4000, 8000)])
    def test_resample(self, orig_freq, new_freq):
        transform = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    @parameterized.expand([("linear",), ("exponential",), ("logarithmic",), ("quarter_sine",), ("half_sine",)])
    def test_fade(self, fade_shape):
        transform = T.Fade(fade_shape=fade_shape)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    @parameterized.expand([(T.TimeMasking,), (T.FrequencyMasking,)])
    def test_masking(self, masking_transform):
        sample_rate = 8000
        n_fft = 400
        spectrogram = get_spectrogram(
            get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2), n_fft=n_fft, power=1
        )
        deterministic_transform = _DeterministicWrapper(masking_transform(400))
        self.assert_grad(deterministic_transform, [spectrogram])

    @parameterized.expand([(T.TimeMasking,), (T.FrequencyMasking,)])
    def test_masking_iid(self, masking_transform):
        sample_rate = 8000
        n_fft = 400
        specs = [
            get_spectrogram(
                get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2, seed=i), n_fft=n_fft, power=1
            )
            for i in range(3)
        ]

        batch = torch.stack(specs)
        assert batch.ndim == 4
        deterministic_transform = _DeterministicWrapper(masking_transform(400, True))
        self.assert_grad(deterministic_transform, [batch])

    def test_time_masking_p(self):
        sample_rate = 8000
        n_fft = 400
        spectrogram = get_spectrogram(
            get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2), n_fft=n_fft, power=1
        )
        time_mask = T.TimeMasking(400, iid_masks=False, p=0.1)
        deterministic_transform = _DeterministicWrapper(time_mask)
        self.assert_grad(deterministic_transform, [spectrogram])

    def test_spectral_centroid(self):
        sample_rate = 8000
        transform = T.SpectralCentroid(sample_rate=sample_rate)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform], nondet_tol=1e-10)

    def test_amplitude_to_db(self):
        sample_rate = 8000
        transform = T.AmplitudeToDB()
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    def test_melscale(self):
        sample_rate = 8000
        n_fft = 400
        n_mels = n_fft // 2 + 1
        transform = T.MelScale(sample_rate=sample_rate, n_mels=n_mels)
        spec = get_spectrogram(
            get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2), n_fft=n_fft, power=1
        )
        self.assert_grad(transform, [spec])

    @parameterized.expand([(1.5, "amplitude"), (2, "power"), (10, "db")])
    def test_vol(self, gain, gain_type):
        sample_rate = 8000
        transform = T.Vol(gain=gain, gain_type=gain_type)
        waveform = get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=2)
        self.assert_grad(transform, [waveform])

    @parameterized.expand(
        [
            ({"cmn_window": 100, "min_cmn_window": 50, "center": False, "norm_vars": False},),
            ({"cmn_window": 100, "min_cmn_window": 50, "center": True, "norm_vars": False},),
            ({"cmn_window": 100, "min_cmn_window": 50, "center": False, "norm_vars": True},),
            ({"cmn_window": 100, "min_cmn_window": 50, "center": True, "norm_vars": True},),
        ]
    )
    def test_sliding_window_cmn(self, kwargs):
        n_fft = 10
        power = 1
        spec = get_spectrogram(get_whitenoise(sample_rate=200, duration=0.05, n_channels=2), n_fft=n_fft, power=power)
        spec_reshaped = spec.transpose(-1, -2)

        transform = T.SlidingWindowCmn(**kwargs)
        self.assert_grad(transform, [spec_reshaped])

    @unittest.expectedFailure
    def test_timestretch_zeros_fail(self):
        """Test that ``T.TimeStretch`` fails gradcheck at 0

        This is because ``F.phase_vocoder`` converts data from cartesian to polar coordinate,
        which performs ``atan2(img, real)``, and gradient is not defined at 0.
        """
        n_fft = 16
        transform = T.TimeStretch(n_freq=n_fft // 2 + 1, fixed_rate=0.99)
        waveform = torch.zeros(2, 40)
        spectrogram = get_spectrogram(waveform, n_fft=n_fft, power=None)
        self.assert_grad(transform, [spectrogram])

    @nested_params([0.7, 0.8, 0.9, 1.0, 1.3])
    def test_timestretch_non_zero(self, rate):
        """Verify that ``T.TimeStretch`` does not fail if it's not close to 0

        ``T.TimeStrech`` is not differentiable around 0, so this test checks the differentiability
        for cases where input is not zero.

        As tested above, when spectrogram contains values close to zero, the gradients are unstable
        and gradcheck fails.

        In this test, we generate spectrogram from random signal, then we push the points around
        zero away from the origin.

        This process does not reflect the real use-case, and it is not practical for users, but
        this helps us understand to what degree the function is differentiable and when not.
        """
        n_fft = 16
        transform = T.TimeStretch(n_freq=n_fft // 2 + 1, fixed_rate=rate)
        waveform = get_whitenoise(sample_rate=40, duration=1, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=n_fft, power=None)

        # 1e-3 is too small (on CPU)
        epsilon = 2e-2
        too_close = spectrogram.abs() < epsilon
        spectrogram[too_close] = epsilon * spectrogram[too_close] / spectrogram[too_close].abs()
        self.assert_grad(transform, [spectrogram])

    def test_psd(self):
        transform = T.PSD()
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=400)
        self.assert_grad(transform, [spectrogram])

    @parameterized.expand(
        [
            [True],
            [False],
        ]
    )
    def test_psd_with_mask(self, multi_mask):
        transform = T.PSD(multi_mask=multi_mask)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=400)
        if multi_mask:
            mask = torch.rand(spectrogram.shape[-3:])
        else:
            mask = torch.rand(spectrogram.shape[-2:])

        self.assert_grad(transform, [spectrogram, mask])

    @parameterized.expand(
        [
            "ref_channel",
            # stv_power and stv_evd test time too long, comment for now
            # "stv_power",
            # "stv_evd",
        ]
    )
    def test_mvdr(self, solution):
        transform = T.MVDR(solution=solution)
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        spectrogram = get_spectrogram(waveform, n_fft=400)
        mask_s = torch.rand(spectrogram.shape[-2:])
        mask_n = torch.rand(spectrogram.shape[-2:])
        self.assert_grad(transform, [spectrogram, mask_s, mask_n])

    def test_rtf_mvdr(self):
        transform = T.RTFMVDR()
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        specgram = get_spectrogram(waveform, n_fft=400)
        channel, freq, _ = specgram.shape
        rtf = torch.rand(freq, channel, dtype=torch.cfloat)
        psd_n = torch.rand(freq, channel, channel, dtype=torch.cfloat)
        reference_channel = 0
        self.assert_grad(transform, [specgram, rtf, psd_n, reference_channel])

    def test_souden_mvdr(self):
        transform = T.SoudenMVDR()
        waveform = get_whitenoise(sample_rate=8000, duration=0.05, n_channels=2)
        specgram = get_spectrogram(waveform, n_fft=400)
        channel, freq, _ = specgram.shape
        psd_s = torch.rand(freq, channel, channel, dtype=torch.cfloat)
        psd_n = torch.rand(freq, channel, channel, dtype=torch.cfloat)
        reference_channel = 0
        self.assert_grad(transform, [specgram, psd_s, psd_n, reference_channel])


class AutogradTestFloat32(TestBaseMixin):
    def assert_grad(
        self,
        transform: torch.nn.Module,
        inputs: List[torch.Tensor],
    ):
        inputs_ = []
        for i in inputs:
            if torch.is_tensor(i):
                i = i.to(dtype=torch.float32, device=self.device)
            inputs_.append(i)
        # gradcheck with float32 requires higher atol and epsilon
        assert gradcheck(transform, inputs, eps=1e-3, atol=1e-3, nondet_tol=0.0)

    @parameterized.expand(
        [
            (rnnt_utils.get_B1_T10_U3_D4_data,),
            (rnnt_utils.get_B2_T4_U3_D3_data,),
            (rnnt_utils.get_B1_T2_U3_D5_data,),
        ]
    )
    def test_rnnt_loss(self, data_func):
        def get_data(data_func, device):
            data = data_func()
            if type(data) == tuple:
                data = data[0]
            return data

        data = get_data(data_func, self.device)
        inputs = (
            data["logits"].to(torch.float32),
            data["targets"],
            data["logit_lengths"],
            data["target_lengths"],
        )
        loss = T.RNNTLoss(blank=data["blank"])

        self.assert_grad(loss, inputs)
