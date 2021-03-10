"""Test defintion common to CPU and CUDA"""
import torch
import torchaudio.functional as F
from parameterized import parameterized

from torchaudio_unittest import common_utils


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
        the filter coefficients is get by `scipy.signal.butter(9, 850, 'hp', fs=22050, output='ba')`
        """
        x = torch.zeros(1024, dtype=self.dtype, device=self.device)
        x[0] = 1
        a = torch.tensor([1., -7.60545606, 25.80187885, -51.23717435,
                          65.62093428, -56.20096964, 32.18274279, -11.88025302,
                          2.56506938, -0.24677075], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.49676025, -4.47084227, 17.88336908, -41.72786118,
                          62.59179178, -62.59179178, 41.72786118, -17.88336908,
                          4.47084227, -0.49676025], dtype=self.dtype, device=self.device)
        y = F.lfilter(x, a, b, False)
        if self.dtype == torch.float64:
            assert torch.all(y.abs() < 0.8)
        else:
            assert torch.any(y.abs() > 1)


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
