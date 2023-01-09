"""Test numerical consistency among single input and batched input."""
import itertools
import math
from functools import partial

import torch
import torchaudio.functional as F
from parameterized import parameterized, parameterized_class
from torchaudio_unittest import common_utils


def _name_from_args(func, _, params):
    """Return a parameterized test name, based on parameter values."""
    return "{}_{}".format(func.__name__, "_".join(str(arg) for arg in params.args))


@parameterized_class(
    [
        # Single-item batch isolates problems that come purely from adding a
        # dimension (rather than processing multiple items)
        {"batch_size": 1},
        {"batch_size": 3},
    ]
)
class TestFunctional(common_utils.TorchaudioTestCase):
    """Test functions defined in `functional` module"""

    backend = "default"

    def assert_batch_consistency(self, functional, inputs, atol=1e-8, rtol=1e-5, seed=42):
        n = inputs[0].size(0)
        for i in range(1, len(inputs)):
            self.assertEqual(inputs[i].size(0), n)
        # Compute items separately, then batch the result
        torch.random.manual_seed(seed)
        items_input = [[ele[i].clone() for ele in inputs] for i in range(n)]
        items_result = torch.stack([functional(*items_input[i]) for i in range(n)])

        # Batch the input and run
        torch.random.manual_seed(seed)
        batch_input = [ele.clone() for ele in inputs]
        batch_result = functional(*batch_input)

        self.assertEqual(items_result, batch_result, rtol=rtol, atol=atol)

    def test_griffinlim(self):
        n_fft = 400
        ws = 400
        hop = 200
        window = torch.hann_window(ws)
        power = 2
        momentum = 0.99
        n_iter = 32
        length = 1000
        batch = torch.rand(self.batch_size, 1, 201, 6)
        kwargs = {
            "window": window,
            "n_fft": n_fft,
            "hop_length": hop,
            "win_length": ws,
            "power": power,
            "n_iter": n_iter,
            "momentum": momentum,
            "length": length,
            "rand_init": False,
        }
        func = partial(F.griffinlim, **kwargs)
        self.assert_batch_consistency(func, inputs=(batch,), atol=5e-5)

    @parameterized.expand(
        list(
            itertools.product(
                [8000, 16000, 44100],
                [1, 2],
            )
        ),
        name_func=_name_from_args,
    )
    def test_detect_pitch_frequency(self, sample_rate, n_channels):
        # Use different frequencies to ensure each item in the batch returns a
        # different answer.
        frequencies = torch.randint(100, 1000, [self.batch_size])
        waveforms = torch.stack(
            [
                common_utils.get_sinusoid(
                    frequency=frequency, sample_rate=sample_rate, n_channels=n_channels, duration=5
                )
                for frequency in frequencies
            ]
        )
        kwargs = {
            "sample_rate": sample_rate,
        }
        func = partial(F.detect_pitch_frequency, **kwargs)
        self.assert_batch_consistency(func, inputs=(waveforms,))

    @parameterized.expand(
        [
            (None,),
            (40.0,),
        ]
    )
    def test_amplitude_to_DB(self, top_db):
        spec = torch.rand(self.batch_size, 2, 100, 100) * 200

        amplitude_mult = 20.0
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        kwargs = {
            "multiplier": amplitude_mult,
            "amin": amin,
            "db_multiplier": db_mult,
            "top_db": top_db,
        }
        func = partial(F.amplitude_to_DB, **kwargs)
        # Test with & without a `top_db` clamp
        self.assert_batch_consistency(func, inputs=(spec,))

    def test_amplitude_to_DB_itemwise_clamps(self):
        """Ensure that the clamps are separate for each spectrogram in a batch.

        The clamp was determined per-batch in a prior implementation, which
        meant it was determined by the loudest item, thus items weren't
        independent. See:

        https://github.com/pytorch/audio/issues/994

        """
        amplitude_mult = 20.0
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 20.0

        # Make a batch of noise
        spec = torch.rand([2, 2, 100, 100]) * 200
        # Make one item blow out the other
        spec[0] += 50
        kwargs = {
            "multiplier": amplitude_mult,
            "amin": amin,
            "db_multiplier": db_mult,
            "top_db": top_db,
        }
        func = partial(F.amplitude_to_DB, **kwargs)
        self.assert_batch_consistency(func, inputs=(spec,))

    def test_amplitude_to_DB_not_channelwise_clamps(self):
        """Check that clamps are applied per-item, not per channel."""
        amplitude_mult = 20.0
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 40.0

        spec = torch.rand([1, 2, 100, 100]) * 200
        # Make one channel blow out the other
        spec[:, 0] += 50

        specwise_dbs = F.amplitude_to_DB(spec, amplitude_mult, amin, db_mult, top_db=top_db)
        channelwise_dbs = torch.stack(
            [F.amplitude_to_DB(spec[:, i], amplitude_mult, amin, db_mult, top_db=top_db) for i in range(spec.size(-3))]
        )

        # Just check channelwise gives a different answer.
        difference = (specwise_dbs - channelwise_dbs).abs()
        assert (difference >= 1e-5).any()

    def test_contrast(self):
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        kwargs = {
            "enhancement_amount": 80.0,
        }
        func = partial(F.contrast, **kwargs)
        self.assert_batch_consistency(func, inputs=(waveforms,))

    def test_dcshift(self):
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        kwargs = {
            "shift": 0.5,
            "limiter_gain": 0.05,
        }
        func = partial(F.dcshift, **kwargs)
        self.assert_batch_consistency(func, inputs=(waveforms,))

    def test_overdrive(self):
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        kwargs = {
            "gain": 45,
            "colour": 30,
        }
        func = partial(F.overdrive, **kwargs)
        self.assert_batch_consistency(func, inputs=(waveforms,))

    def test_phaser(self):
        sample_rate = 8000
        n_channels = 2
        waveform = common_utils.get_whitenoise(
            sample_rate=sample_rate, n_channels=self.batch_size * n_channels, duration=1
        )
        batch = waveform.view(self.batch_size, n_channels, waveform.size(-1))
        kwargs = {
            "sample_rate": sample_rate,
        }
        func = partial(F.phaser, **kwargs)
        self.assert_batch_consistency(func, inputs=(batch,))

    def test_flanger(self):
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        sample_rate = 8000
        kwargs = {
            "sample_rate": sample_rate,
        }
        func = partial(F.flanger, **kwargs)
        self.assert_batch_consistency(func, inputs=(waveforms,))

    @parameterized.expand(
        list(
            itertools.product(
                [True, False],  # center
                [True, False],  # norm_vars
            )
        ),
        name_func=_name_from_args,
    )
    def test_sliding_window_cmn(self, center, norm_vars):
        spectrogram = torch.rand(self.batch_size, 2, 1024, 1024) * 200
        kwargs = {
            "center": center,
            "norm_vars": norm_vars,
        }
        func = partial(F.sliding_window_cmn, **kwargs)
        self.assert_batch_consistency(func, inputs=(spectrogram,))

    @parameterized.expand([("sinc_interp_hann"), ("sinc_interp_kaiser")])
    def test_resample_waveform(self, resampling_method):
        num_channels = 3
        sr = 16000
        new_sr = sr // 2
        multi_sound = common_utils.get_whitenoise(
            sample_rate=sr,
            n_channels=num_channels,
            duration=0.5,
        )
        kwargs = {
            "orig_freq": sr,
            "new_freq": new_sr,
            "resampling_method": resampling_method,
        }
        func = partial(F.resample, **kwargs)

        self.assert_batch_consistency(
            func,
            inputs=(multi_sound,),
            rtol=1e-4,
            atol=1e-7,
        )

    @common_utils.skipIfNoKaldi
    def test_compute_kaldi_pitch(self):
        sample_rate = 44100
        n_channels = 2
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, n_channels=self.batch_size * n_channels)
        batch = waveform.view(self.batch_size, n_channels, waveform.size(-1))
        kwargs = {
            "sample_rate": sample_rate,
        }
        func = partial(F.compute_kaldi_pitch, **kwargs)
        self.assert_batch_consistency(func, inputs=(batch,))

    def test_lfilter(self):
        signal_length = 2048
        x = torch.randn(self.batch_size, signal_length)
        a = torch.rand(self.batch_size, 3)
        b = torch.rand(self.batch_size, 3)
        self.assert_batch_consistency(F.lfilter, inputs=(x, a, b))

    def test_filtfilt(self):
        signal_length = 2048
        x = torch.randn(self.batch_size, signal_length)
        a = torch.rand(self.batch_size, 3)
        b = torch.rand(self.batch_size, 3)
        self.assert_batch_consistency(F.filtfilt, inputs=(x, a, b))

    def test_psd(self):
        batch_size = 2
        channel = 3
        sample_rate = 44100
        n_fft = 400
        n_fft_bin = 201
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=batch_size * channel)
        specgram = common_utils.get_spectrogram(waveform, n_fft=n_fft, hop_length=100)
        specgram = specgram.view(batch_size, channel, n_fft_bin, specgram.size(-1))
        self.assert_batch_consistency(F.psd, (specgram,))

    def test_psd_with_mask(self):
        batch_size = 2
        channel = 3
        sample_rate = 44100
        n_fft = 400
        n_fft_bin = 201
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, duration=0.05, n_channels=batch_size * channel)
        specgram = common_utils.get_spectrogram(waveform, n_fft=n_fft, hop_length=100)
        specgram = specgram.view(batch_size, channel, n_fft_bin, specgram.size(-1))
        mask = torch.rand(batch_size, n_fft_bin, specgram.size(-1))
        self.assert_batch_consistency(F.psd, (specgram, mask))

    def test_mvdr_weights_souden(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 10
        psd_speech = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        kwargs = {
            "reference_channel": 0,
        }
        func = partial(F.mvdr_weights_souden, **kwargs)
        self.assert_batch_consistency(func, (psd_noise, psd_speech))

    def test_mvdr_weights_souden_with_tensor(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 10
        psd_speech = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        reference_channel = torch.zeros(batch_size, channel)
        reference_channel[..., 0].fill_(1)
        self.assert_batch_consistency(F.mvdr_weights_souden, (psd_noise, psd_speech, reference_channel))

    def test_mvdr_weights_rtf(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 129
        rtf = torch.rand(batch_size, n_fft_bin, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        kwargs = {
            "reference_channel": 0,
        }
        func = partial(F.mvdr_weights_rtf, **kwargs)
        self.assert_batch_consistency(func, (rtf, psd_noise))

    def test_mvdr_weights_rtf_with_tensor(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 129
        rtf = torch.rand(batch_size, n_fft_bin, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        reference_channel = torch.zeros(batch_size, channel)
        reference_channel[..., 0].fill_(1)
        self.assert_batch_consistency(F.mvdr_weights_rtf, (rtf, psd_noise, reference_channel))

    def test_rtf_evd(self):
        batch_size = 2
        channel = 4
        n_fft_bin = 5
        spectrum = torch.rand(batch_size, n_fft_bin, channel, dtype=torch.cfloat)
        psd = torch.einsum("...c,...d->...cd", spectrum, spectrum.conj())
        self.assert_batch_consistency(F.rtf_evd, (psd,))

    @parameterized.expand(
        [
            (1,),
            (3,),
        ]
    )
    def test_rtf_power(self, n_iter):
        channel = 4
        batch_size = 2
        n_fft_bin = 10
        psd_speech = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        kwargs = {
            "reference_channel": 0,
            "n_iter": n_iter,
        }
        func = partial(F.rtf_power, **kwargs)
        self.assert_batch_consistency(func, (psd_speech, psd_noise))

    @parameterized.expand(
        [
            (1,),
            (3,),
        ]
    )
    def test_rtf_power_with_tensor(self, n_iter):
        channel = 4
        batch_size = 2
        n_fft_bin = 10
        psd_speech = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        psd_noise = torch.rand(batch_size, n_fft_bin, channel, channel, dtype=torch.cfloat)
        reference_channel = torch.zeros(batch_size, channel)
        reference_channel[..., 0].fill_(1)
        kwargs = {
            "n_iter": n_iter,
        }
        func = partial(F.rtf_power, **kwargs)
        self.assert_batch_consistency(func, (psd_speech, psd_noise, reference_channel))

    def test_apply_beamforming(self):
        sr = 8000
        n_fft = 400
        batch_size, num_channels = 2, 3
        n_fft_bin = n_fft // 2 + 1
        x = common_utils.get_whitenoise(sample_rate=sr, duration=0.05, n_channels=batch_size * num_channels)
        specgram = common_utils.get_spectrogram(x, n_fft=n_fft, hop_length=100)
        specgram = specgram.view(batch_size, num_channels, n_fft_bin, specgram.size(-1))
        beamform_weights = torch.rand(batch_size, n_fft_bin, num_channels, dtype=torch.cfloat)
        self.assert_batch_consistency(F.apply_beamforming, (beamform_weights, specgram))
