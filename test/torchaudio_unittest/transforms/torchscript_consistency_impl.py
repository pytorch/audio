"""Test suites for jit-ability and its numerical compatibility"""

import torch
import torchaudio.transforms as T
from parameterized import parameterized
from torchaudio_unittest import common_utils
from torchaudio_unittest.common_utils import skipIfRocm, TestBaseMixin, torch_script


class Transforms(TestBaseMixin):
    """Implements test for Transforms that are performed for different devices"""

    def _assert_consistency(self, transform, tensor, *args):
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        transform = transform.to(device=self.device, dtype=self.dtype)

        ts_transform = torch_script(transform)

        output = transform(tensor, *args)
        ts_output = ts_transform(tensor, *args)
        self.assertEqual(ts_output, output)

    def _assert_consistency_complex(self, transform, tensor, *args):
        assert tensor.is_complex()
        tensor = tensor.to(device=self.device, dtype=self.complex_dtype)
        transform = transform.to(device=self.device, dtype=self.dtype)

        ts_transform = torch_script(transform)

        output = transform(tensor, *args)
        ts_output = ts_transform(tensor, *args)
        self.assertEqual(ts_output, output)

    def test_Spectrogram(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.Spectrogram(), tensor)

    def test_Spectrogram_return_complex(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.Spectrogram(power=None, return_complex=True), tensor)

    def test_InverseSpectrogram(self):
        tensor = common_utils.get_whitenoise(sample_rate=8000)
        spectrogram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        self._assert_consistency_complex(T.InverseSpectrogram(n_fft=400, hop_length=100), spectrogram)

    @skipIfRocm
    def test_GriffinLim(self):
        tensor = torch.rand((1, 201, 6))
        self._assert_consistency(T.GriffinLim(length=1000, rand_init=False), tensor)

    def test_AmplitudeToDB(self):
        spec = torch.rand((6, 201))
        self._assert_consistency(T.AmplitudeToDB(), spec)

    def test_MelScale(self):
        spec_f = torch.rand((1, 201, 6))
        self._assert_consistency(T.MelScale(n_stft=201), spec_f)

    def test_MelSpectrogram(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.MelSpectrogram(), tensor)

    def test_MFCC(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.MFCC(), tensor)

    def test_LFCC(self):
        tensor = torch.rand((1, 1000))
        self._assert_consistency(T.LFCC(), tensor)

    def test_Resample(self):
        sr1, sr2 = 16000, 8000
        tensor = common_utils.get_whitenoise(sample_rate=sr1)
        self._assert_consistency(T.Resample(sr1, sr2), tensor)

    def test_MuLawEncoding(self):
        tensor = common_utils.get_whitenoise()
        self._assert_consistency(T.MuLawEncoding(), tensor)

    def test_MuLawDecoding(self):
        tensor = torch.rand((1, 10))
        self._assert_consistency(T.MuLawDecoding(), tensor)

    def test_ComputeDelta(self):
        tensor = torch.rand((1, 10))
        self._assert_consistency(T.ComputeDeltas(), tensor)

    def test_Fade(self):
        waveform = common_utils.get_whitenoise()
        fade_in_len = 3000
        fade_out_len = 3000
        self._assert_consistency(T.Fade(fade_in_len, fade_out_len), waveform)

    def test_FrequencyMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        self._assert_consistency(T.FrequencyMasking(freq_mask_param=60, iid_masks=False), tensor)

    def test_TimeMasking(self):
        tensor = torch.rand((10, 2, 50, 10, 2))
        self._assert_consistency(T.TimeMasking(time_mask_param=30, iid_masks=False), tensor)

    def test_Vol(self):
        waveform = common_utils.get_whitenoise()
        self._assert_consistency(T.Vol(1.1), waveform)

    def test_SlidingWindowCmn(self):
        tensor = torch.rand((1000, 10))
        self._assert_consistency(T.SlidingWindowCmn(), tensor)

    def test_Vad(self):
        filepath = common_utils.get_asset_path("vad-go-mono-32000.wav")
        waveform, sample_rate = common_utils.load_wav(filepath)
        self._assert_consistency(T.Vad(sample_rate=sample_rate), waveform)

    def test_SpectralCentroid(self):
        sample_rate = 44100
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate)
        self._assert_consistency(T.SpectralCentroid(sample_rate=sample_rate), waveform)

    def test_TimeStretch(self):
        n_fft = 1025
        n_freq = n_fft // 2 + 1
        hop_length = 512
        fixed_rate = 1.3
        tensor = torch.rand((10, 2, n_freq, 10), dtype=torch.cfloat)
        batch = 10
        num_channels = 2

        waveform = common_utils.get_whitenoise(sample_rate=8000, n_channels=batch * num_channels)
        tensor = common_utils.get_spectrogram(waveform, n_fft=n_fft)
        tensor = tensor.reshape(batch, num_channels, n_freq, -1)
        self._assert_consistency_complex(
            T.TimeStretch(n_freq=n_freq, hop_length=hop_length, fixed_rate=fixed_rate),
            tensor,
        )

    def test_PitchShift(self):
        sample_rate = 8000
        n_steps = 4
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate)
        pitch_shift = T.PitchShift(sample_rate=sample_rate, n_steps=n_steps)
        # dry-run for initializing parameters
        pitch_shift(waveform)
        self._assert_consistency(pitch_shift, waveform)

    def test_PSD(self):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        spectrogram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        spectrogram = spectrogram.to(self.device)
        self._assert_consistency_complex(T.PSD(), spectrogram)

    def test_PSD_with_mask(self):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        spectrogram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        spectrogram = spectrogram.to(self.device)
        mask = torch.rand(spectrogram.shape[-2:], device=self.device)
        self._assert_consistency_complex(T.PSD(), spectrogram, mask)

    @parameterized.expand(
        [
            ["ref_channel", True],
            ["stv_evd", True],
            ["stv_power", True],
            ["ref_channel", False],
            ["stv_evd", False],
            ["stv_power", False],
        ]
    )
    def test_MVDR(self, solution, online):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        spectrogram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        mask_s = torch.rand(spectrogram.shape[-2:], device=self.device)
        mask_n = torch.rand(spectrogram.shape[-2:], device=self.device)
        self._assert_consistency_complex(T.MVDR(solution=solution, online=online), spectrogram, mask_s, mask_n)

    def test_rtf_mvdr(self):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        specgram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        channel, freq, _ = specgram.shape
        rtf = torch.rand(freq, channel, dtype=self.complex_dtype, device=self.device)
        psd_n = torch.rand(freq, channel, channel, dtype=self.complex_dtype, device=self.device)
        reference_channel = 0
        self._assert_consistency_complex(T.RTFMVDR(), specgram, rtf, psd_n, reference_channel)

    def test_souden_mvdr(self):
        tensor = common_utils.get_whitenoise(sample_rate=8000, n_channels=4)
        specgram = common_utils.get_spectrogram(tensor, n_fft=400, hop_length=100)
        channel, freq, _ = specgram.shape
        psd_s = torch.rand(freq, channel, channel, dtype=self.complex_dtype, device=self.device)
        psd_n = torch.rand(freq, channel, channel, dtype=self.complex_dtype, device=self.device)
        reference_channel = 0
        self._assert_consistency_complex(T.SoudenMVDR(), specgram, psd_s, psd_n, reference_channel)

    @common_utils.nested_params(
        ["Convolve", "FFTConvolve"],
        ["full", "valid", "same"],
    )
    def test_convolve(self, cls, mode):
        leading_dims = (2, 3, 2)
        L_x, L_y = 32, 55
        x = torch.rand(*leading_dims, L_x, dtype=self.dtype, device=self.device)
        y = torch.rand(*leading_dims, L_y, dtype=self.dtype, device=self.device)

        convolve = getattr(T, cls)(mode=mode).to(device=self.device, dtype=self.dtype)
        output = convolve(x, y)
        ts_output = torch_script(convolve)(x, y)
        self.assertEqual(ts_output, output)

    @common_utils.nested_params([True, False])
    def test_speed(self, use_lengths):
        leading_dims = (3, 2)
        time = 200
        waveform = torch.rand(*leading_dims, time, dtype=self.dtype, device=self.device, requires_grad=True)

        if use_lengths:
            lengths = torch.randint(1, time, leading_dims, dtype=self.dtype, device=self.device)
        else:
            lengths = None

        speed = T.Speed(1000, 0.9).to(self.device, self.dtype)
        output = speed(waveform, lengths)
        ts_output = torch_script(speed)(waveform, lengths)
        self.assertEqual(ts_output, output)

    @common_utils.nested_params([True, False])
    def test_speed_perturbation(self, use_lengths):
        leading_dims = (3, 2)
        time = 200
        waveform = torch.rand(*leading_dims, time, dtype=self.dtype, device=self.device, requires_grad=True)

        if use_lengths:
            lengths = torch.randint(1, time, leading_dims, dtype=self.dtype, device=self.device)
        else:
            lengths = None

        speed = T.SpeedPerturbation(1000, [0.9]).to(self.device, self.dtype)
        output = speed(waveform, lengths)
        ts_output = torch_script(speed)(waveform, lengths)
        self.assertEqual(ts_output, output)

    @common_utils.nested_params([True, False])
    def test_add_noise(self, use_lengths):
        leading_dims = (2, 3)
        L = 31

        waveform = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        noise = torch.rand(*leading_dims, L, dtype=self.dtype, device=self.device, requires_grad=True)
        if use_lengths:
            lengths = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True)
        else:
            lengths = None
        snr = torch.rand(*leading_dims, dtype=self.dtype, device=self.device, requires_grad=True) * 10

        add_noise = T.AddNoise().to(self.device, self.dtype)
        output = add_noise(waveform, noise, snr, lengths)
        ts_output = torch_script(add_noise)(waveform, noise, snr, lengths)
        self.assertEqual(ts_output, output)

    def test_preemphasis(self):
        waveform = torch.rand(3, 4, 10, dtype=self.dtype, device=self.device)
        preemphasis = T.Preemphasis(coeff=0.97).to(dtype=self.dtype, device=self.device)
        output = preemphasis(waveform)
        ts_output = torch_script(preemphasis)(waveform)
        self.assertEqual(ts_output, output)

    def test_deemphasis(self):
        waveform = torch.rand(3, 4, 10, dtype=self.dtype, device=self.device)
        deemphasis = T.Deemphasis(coeff=0.97).to(dtype=self.dtype, device=self.device)
        output = deemphasis(waveform)
        ts_output = torch_script(deemphasis)(waveform)
        self.assertEqual(ts_output, output)


class TransformsFloat32Only(TestBaseMixin):
    def test_rnnt_loss(self):
        logits = torch.tensor(
            [
                [
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.2, 0.8, 0.1]],
                    [[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1], [0.7, 0.1, 0.2, 0.1, 0.1]],
                ]
            ]
        )
        tensor = logits.to(device=self.device, dtype=torch.float32)
        targets = torch.tensor([[1, 2]], device=tensor.device, dtype=torch.int32)
        logit_lengths = torch.tensor([2], device=tensor.device, dtype=torch.int32)
        target_lengths = torch.tensor([2], device=tensor.device, dtype=torch.int32)

        self._assert_consistency(T.RNNTLoss(), logits, targets, logit_lengths, target_lengths)
