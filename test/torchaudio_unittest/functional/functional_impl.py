"""Test defintion common to CPU and CUDA"""
import torch
import torchaudio.functional as F
from parameterized import parameterized
import numpy as np
from scipy import signal

from torchaudio_unittest import common_utils
from torchaudio_unittest.common_utils import nested_params


class Lfilter(common_utils.TestBaseMixin):
    def test_simple(self):
        """
        Create a very basic signal,
        Then make a simple 4th order delay
        The output should be same as the input but shifted
        """

        torch.random.manual_seed(42)
        waveform = torch.rand(2, 44100 * 1, dtype=self.dtype, device=self.device)
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=self.dtype, device=self.device)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)

        self.assertEqual(output_waveform[:, 3:], waveform[:, 0:-3], atol=1e-5, rtol=1e-5)

    def test_clamp(self):
        input_signal = torch.ones(1, 44100 * 1, dtype=self.dtype, device=self.device)
        b_coeffs = torch.tensor([1, 0], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, -0.95], dtype=self.dtype, device=self.device)
        output_signal = F.lfilter(input_signal, a_coeffs, b_coeffs, clamp=True)
        assert output_signal.max() <= 1
        output_signal = F.lfilter(input_signal, a_coeffs, b_coeffs, clamp=False)
        assert output_signal.max() > 1

    @parameterized.expand([
        ((44100,),),
        ((3, 44100),),
        ((2, 3, 44100),),
        ((1, 2, 3, 44100),)
    ])
    def test_shape(self, shape):
        torch.random.manual_seed(42)
        waveform = torch.rand(*shape, dtype=self.dtype, device=self.device)
        b_coeffs = torch.tensor([0, 0, 0, 1], dtype=self.dtype, device=self.device)
        a_coeffs = torch.tensor([1, 0, 0, 0], dtype=self.dtype, device=self.device)
        output_waveform = F.lfilter(waveform, a_coeffs, b_coeffs)
        assert shape == waveform.size() == output_waveform.size()

    def test_9th_order_filter_stability(self):
        """
        Validate the precision of lfilter against reference scipy implementation when using high order filter.
        The reference implementation use cascaded second-order filters so is more numerically accurate.
        """
        # create an impulse signal
        x = torch.zeros(1024, dtype=self.dtype, device=self.device)
        x[0] = 1

        # get target impulse response
        sos = signal.butter(9, 850, 'hp', fs=22050, output='sos')
        y = torch.from_numpy(signal.sosfilt(sos, x.cpu().numpy())).to(self.dtype).to(self.device)

        # get lfilter coefficients
        b, a = signal.butter(9, 850, 'hp', fs=22050, output='ba')
        b, a = torch.from_numpy(b).to(self.dtype).to(self.device), torch.from_numpy(
            a).to(self.dtype).to(self.device)

        # predict impulse response
        yhat = F.lfilter(x, a, b, False)
        self.assertEqual(yhat, y, atol=1e-4, rtol=1e-5)


class Spectrogram(common_utils.TestBaseMixin):
    @parameterized.expand([(0., ), (1., ), (2., ), (3., )])
    def test_grad_at_zero(self, power):
        """The gradient of power spectrogram should not be nan but zero near x=0

        https://github.com/pytorch/audio/issues/993
        """
        x = torch.zeros(1, 22050, requires_grad=True)
        spec = F.spectrogram(
            x,
            pad=0,
            window=None,
            n_fft=2048,
            hop_length=None,
            win_length=None,
            power=power,
            normalized=False,
        )
        spec.sum().backward()
        assert not x.grad.isnan().sum()


class FunctionalComplex(common_utils.TestBaseMixin):
    complex_dtype = None
    real_dtype = None
    device = None

    @nested_params(
        [0.5, 1.01, 1.3],
        [True, False],
    )
    def test_phase_vocoder_shape(self, rate, test_pseudo_complex):
        hop_length = 256
        num_freq = 1025
        num_frames = 400
        batch_size = 2

        torch.random.manual_seed(42)
        spec = torch.randn(
            batch_size, num_freq, num_frames, dtype=self.complex_dtype, device=self.device)
        if test_pseudo_complex:
            spec = torch.view_as_real(spec)

        phase_advance = torch.linspace(
            0,
            np.pi * hop_length,
            num_freq,
            dtype=self.real_dtype, device=self.device)[..., None]

        spec_stretch = F.phase_vocoder(spec, rate=rate, phase_advance=phase_advance)

        assert spec.dim() == spec_stretch.dim()
        expected_shape = torch.Size([batch_size, num_freq, int(np.ceil(num_frames / rate))])
        output_shape = (torch.view_as_complex(spec_stretch) if test_pseudo_complex else spec_stretch).shape
        assert output_shape == expected_shape
