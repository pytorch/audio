"""Test numerical consistency among single input and batched input."""
import itertools
import math

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

    def assert_batch_consistency(self, functional, batch, *args, atol=1e-8, rtol=1e-5, seed=42, **kwargs):
        n = batch.size(0)

        # Compute items separately, then batch the result
        torch.random.manual_seed(seed)
        items_input = batch.clone()
        items_result = torch.stack([functional(items_input[i], *args, **kwargs) for i in range(n)])

        # Batch the input and run
        torch.random.manual_seed(seed)
        batch_input = batch.clone()
        batch_result = functional(batch_input, *args, **kwargs)

        self.assertEqual(items_input, batch_input, rtol=rtol, atol=atol)
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
        torch.random.manual_seed(0)
        batch = torch.rand(self.batch_size, 1, 201, 6)
        self.assert_batch_consistency(
            F.griffinlim, batch, window, n_fft, hop, ws, power, n_iter, momentum, length, 0, atol=5e-5
        )

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
        torch.manual_seed(0)
        frequencies = torch.randint(100, 1000, [self.batch_size])
        waveforms = torch.stack(
            [
                common_utils.get_sinusoid(
                    frequency=frequency, sample_rate=sample_rate, n_channels=n_channels, duration=5
                )
                for frequency in frequencies
            ]
        )
        self.assert_batch_consistency(F.detect_pitch_frequency, waveforms, sample_rate)

    def test_amplitude_to_DB(self):
        torch.manual_seed(0)
        spec = torch.rand(self.batch_size, 2, 100, 100) * 200

        amplitude_mult = 20.0
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))

        # Test with & without a `top_db` clamp
        self.assert_batch_consistency(F.amplitude_to_DB, spec, amplitude_mult, amin, db_mult, top_db=None)
        self.assert_batch_consistency(F.amplitude_to_DB, spec, amplitude_mult, amin, db_mult, top_db=40.0)

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
        torch.manual_seed(0)
        spec = torch.rand([2, 2, 100, 100]) * 200
        # Make one item blow out the other
        spec[0] += 50

        batchwise_dbs = F.amplitude_to_DB(spec, amplitude_mult, amin, db_mult, top_db=top_db)
        itemwise_dbs = torch.stack(
            [F.amplitude_to_DB(item, amplitude_mult, amin, db_mult, top_db=top_db) for item in spec]
        )

        self.assertEqual(batchwise_dbs, itemwise_dbs)

    def test_amplitude_to_DB_not_channelwise_clamps(self):
        """Check that clamps are applied per-item, not per channel."""
        amplitude_mult = 20.0
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 40.0

        torch.manual_seed(0)
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
        torch.random.manual_seed(0)
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        self.assert_batch_consistency(F.contrast, waveforms, enhancement_amount=80.0)

    def test_dcshift(self):
        torch.random.manual_seed(0)
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        self.assert_batch_consistency(F.dcshift, waveforms, shift=0.5, limiter_gain=0.05)

    def test_overdrive(self):
        torch.random.manual_seed(0)
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        self.assert_batch_consistency(F.overdrive, waveforms, gain=45, colour=30)

    def test_phaser(self):
        sample_rate = 44100
        n_channels = 2
        waveform = common_utils.get_whitenoise(
            sample_rate=sample_rate, n_channels=self.batch_size * n_channels, duration=1
        )
        batch = waveform.view(self.batch_size, n_channels, waveform.size(-1))
        self.assert_batch_consistency(F.phaser, batch, sample_rate)

    def test_flanger(self):
        torch.random.manual_seed(0)
        waveforms = torch.rand(self.batch_size, 2, 100) - 0.5
        sample_rate = 44100
        self.assert_batch_consistency(F.flanger, waveforms, sample_rate)

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
        torch.manual_seed(0)
        spectrogram = torch.rand(self.batch_size, 2, 1024, 1024) * 200
        self.assert_batch_consistency(F.sliding_window_cmn, spectrogram, center=center, norm_vars=norm_vars)

    @parameterized.expand([("sinc_interpolation"), ("kaiser_window")])
    def test_resample_waveform(self, resampling_method):
        num_channels = 3
        sr = 16000
        new_sr = sr // 2
        multi_sound = common_utils.get_whitenoise(
            sample_rate=sr,
            n_channels=num_channels,
            duration=0.5,
        )

        self.assert_batch_consistency(
            F.resample,
            multi_sound,
            orig_freq=sr,
            new_freq=new_sr,
            resampling_method=resampling_method,
            rtol=1e-4,
            atol=1e-7,
        )

    @common_utils.skipIfNoKaldi
    def test_compute_kaldi_pitch(self):
        sample_rate = 44100
        n_channels = 2
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate, n_channels=self.batch_size * n_channels)
        batch = waveform.view(self.batch_size, n_channels, waveform.size(-1))
        self.assert_batch_consistency(F.compute_kaldi_pitch, batch, sample_rate=sample_rate)

    def test_lfilter(self):
        signal_length = 2048
        torch.manual_seed(2434)
        x = torch.randn(self.batch_size, signal_length)
        a = torch.rand(self.batch_size, 3)
        b = torch.rand(self.batch_size, 3)

        batchwise_output = F.lfilter(x, a, b, batching=True)
        itemwise_output = torch.stack([F.lfilter(x[i], a[i], b[i]) for i in range(self.batch_size)])

        self.assertEqual(batchwise_output, itemwise_output)

    def test_filtfilt(self):
        signal_length = 2048
        torch.manual_seed(2434)
        x = torch.randn(self.batch_size, signal_length)
        a = torch.rand(self.batch_size, 3)
        b = torch.rand(self.batch_size, 3)

        batchwise_output = F.filtfilt(x, a, b)
        itemwise_output = torch.stack([F.filtfilt(x[i], a[i], b[i]) for i in range(self.batch_size)])

        self.assertEqual(batchwise_output, itemwise_output)
