import torch
import torchaudio.functional as F
from torch.autograd import gradcheck
from torchaudio_unittest import common_utils


class Autograd(common_utils.TestBaseMixin):
    def test_x_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        x.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_a_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        a.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_b_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        b.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_all_grad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(2, 4, 256 * 2, dtype=self.dtype, device=self.device)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device)
        b.requires_grad = True
        a.requires_grad = True
        x.requires_grad = True
        assert gradcheck(F.lfilter, (x, a, b), eps=1e-10)

    def test_biquad(self):
        torch.random.manual_seed(2434)
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        a = torch.tensor([0.7, 0.2, 0.6], dtype=self.dtype, device=self.device, requires_grad=True)
        b = torch.tensor([0.4, 0.2, 0.9], dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.biquad, (x, b[0], b[1], b[2], a[0], a[1], a[2]), eps=1e-10)

    def test_band_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.band_biquad, (x, sr, central_freq, Q))

    def test_band_biquad_with_noise(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.band_biquad, (x, sr, central_freq, Q, True))

    def test_bass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(100, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        gain = torch.tensor(10, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.bass_biquad, (x, sr, gain, central_freq, Q))

    def test_treble_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(3000, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        gain = torch.tensor(10, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.treble_biquad, (x, sr, gain, central_freq, Q))

    def test_allpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.allpass_biquad, (x, sr, central_freq, Q))

    def test_lowpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        cutoff_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.lowpass_biquad, (x, sr, cutoff_freq, Q))

    def test_highpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        cutoff_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.highpass_biquad, (x, sr, cutoff_freq, Q))

    def test_bandpass_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.bandpass_biquad, (x, sr, central_freq, Q))

    def test_bandpass_biquad_with_const_skirt_gain(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.bandpass_biquad, (x, sr, central_freq, Q, True))

    def test_equalizer_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        gain = torch.tensor(10, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.equalizer_biquad, (x, sr, central_freq, gain, Q))

    def test_bandreject_biquad(self):
        torch.random.manual_seed(2434)
        sr = 22050
        x = torch.rand(1024, dtype=self.dtype, device=self.device, requires_grad=True)
        central_freq = torch.tensor(800, dtype=self.dtype, device=self.device, requires_grad=True)
        Q = torch.tensor(0.7, dtype=self.dtype, device=self.device, requires_grad=True)
        assert gradcheck(F.bandreject_biquad, (x, sr, central_freq, Q))
