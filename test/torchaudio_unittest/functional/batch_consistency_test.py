"""Test numerical consistency among single input and batched input."""
import itertools
import math

from parameterized import parameterized
import torch
import torchaudio
import torchaudio.functional as F

from torchaudio_unittest import common_utils


class TestFunctional(common_utils.TorchaudioTestCase):
    backend = 'default'
    """Test functions defined in `functional` module"""
    def assert_batch_consistency(
            self, functional, tensor, *args, batch_size=1, atol=1e-8,
            rtol=1e-5, seed=42, **kwargs):
        # run then batch the result
        torch.random.manual_seed(seed)
        expected = functional(tensor.clone(), *args, **kwargs)
        expected = expected.repeat([batch_size] + [1] * expected.dim())

        # batch the input and run
        torch.random.manual_seed(seed)
        pattern = [batch_size] + [1] * tensor.dim()
        computed = functional(tensor.repeat(pattern), *args, **kwargs)

        self.assertEqual(computed, expected, rtol=rtol, atol=atol)

    def assert_batch_consistencies(
            self, functional, tensor, *args, atol=1e-8, rtol=1e-5,
            seed=42, **kwargs):
        self.assert_batch_consistency(
            functional, tensor, *args, batch_size=1, atol=atol,
            rtol=rtol, seed=seed, **kwargs)
        self.assert_batch_consistency(
            functional, tensor, *args, batch_size=3, atol=atol,
            rtol=rtol, seed=seed, **kwargs)

    def test_griffinlim(self):
        n_fft = 400
        ws = 400
        hop = 200
        window = torch.hann_window(ws)
        power = 2
        normalize = False
        momentum = 0.99
        n_iter = 32
        length = 1000
        tensor = torch.rand((1, 201, 6))
        self.assert_batch_consistencies(
            F.griffinlim, tensor, window, n_fft, hop, ws, power, normalize,
            n_iter, momentum, length, 0, atol=5e-5)

    @parameterized.expand(list(itertools.product(
        [100, 440],
        [8000, 16000, 44100],
        [1, 2],
    )), name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}')
    def test_detect_pitch_frequency(self, frequency, sample_rate, n_channels):
        waveform = common_utils.get_sinusoid(
            frequency=frequency, sample_rate=sample_rate,
            n_channels=n_channels, duration=5)
        self.assert_batch_consistencies(
            F.detect_pitch_frequency, waveform, sample_rate)

    def test_amplitude_to_DB(self):
        torch.manual_seed(0)
        spec = torch.rand(2, 100, 100) * 200

        amplitude_mult = 20.
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))

        # Test with & without a `top_db` clamp
        self.assert_batch_consistencies(
            F.amplitude_to_DB, spec, amplitude_mult,
            amin, db_mult, top_db=None)
        self.assert_batch_consistencies(
            F.amplitude_to_DB, spec, amplitude_mult,
            amin, db_mult, top_db=40.)

    def test_amplitude_to_DB_itemwise_clamps(self):
        """Ensure that the clamps are separate for each spectrogram in a batch.

        The clamp was determined per-batch in a prior implementation, which
        meant it was determined by the loudest item, thus items weren't
        independent. See:

        https://github.com/pytorch/audio/issues/994

        """
        amplitude_mult = 20.
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 20.

        # Make a batch of noise
        torch.manual_seed(0)
        spec = torch.rand([2, 2, 100, 100]) * 200
        # Make one item blow out the other
        spec[0] += 50

        batchwise_dbs = F.amplitude_to_DB(spec, amplitude_mult, amin,
                                          db_mult, top_db=top_db)
        itemwise_dbs = torch.stack([
            F.amplitude_to_DB(item, amplitude_mult, amin,
                              db_mult, top_db=top_db)
            for item in spec
        ])

        self.assertEqual(batchwise_dbs, itemwise_dbs)

    def test_amplitude_to_DB_not_channelwise_clamps(self):
        """Check that clamps are applied per-item, not per channel."""
        amplitude_mult = 20.
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 40.

        torch.manual_seed(0)
        spec = torch.rand([1, 2, 100, 100]) * 200
        # Make one channel blow out the other
        spec[:, 0] += 50

        specwise_dbs = F.amplitude_to_DB(spec, amplitude_mult, amin,
                                         db_mult, top_db=top_db)
        channelwise_dbs = torch.stack([
            F.amplitude_to_DB(spec[:, i], amplitude_mult, amin,
                              db_mult, top_db=top_db)
            for i in range(spec.size(-3))
        ])

        # Just check channelwise gives a different answer.
        difference = (specwise_dbs - channelwise_dbs).abs()
        assert (difference >= 1e-5).any()

    def test_contrast(self):
        waveform = torch.rand(2, 100) - 0.5
        self.assert_batch_consistencies(
            F.contrast, waveform, enhancement_amount=80.)

    def test_dcshift(self):
        waveform = torch.rand(2, 100) - 0.5
        self.assert_batch_consistencies(
            F.dcshift, waveform, shift=0.5, limiter_gain=0.05)

    def test_overdrive(self):
        waveform = torch.rand(2, 100) - 0.5
        self.assert_batch_consistencies(
            F.overdrive, waveform, gain=45, colour=30)

    def test_phaser(self):
        sample_rate = 44100
        waveform = common_utils.get_whitenoise(
            sample_rate=sample_rate, duration=5,
        )
        self.assert_batch_consistencies(F.phaser, waveform, sample_rate)

    def test_flanger(self):
        torch.random.manual_seed(40)
        waveform = torch.rand(2, 100) - 0.5
        sample_rate = 44100
        self.assert_batch_consistencies(F.flanger, waveform, sample_rate)

    def test_sliding_window_cmn(self):
        waveform = torch.randn(2, 1024) - 0.5
        self.assert_batch_consistencies(
            F.sliding_window_cmn, waveform, center=True, norm_vars=True)
        self.assert_batch_consistencies(
            F.sliding_window_cmn, waveform, center=True, norm_vars=False)
        self.assert_batch_consistencies(
            F.sliding_window_cmn, waveform, center=False, norm_vars=True)
        self.assert_batch_consistencies(
            F.sliding_window_cmn, waveform, center=False, norm_vars=False)

    def test_vad(self):
        common_utils.set_audio_backend('default')
        filepath = common_utils.get_asset_path("vad-go-mono-32000.wav")
        waveform, sample_rate = torchaudio.load(filepath)
        self.assert_batch_consistencies(
            F.vad, waveform, sample_rate=sample_rate)

    @common_utils.skipIfNoExtension
    def test_compute_kaldi_pitch(self):
        sample_rate = 44100
        waveform = common_utils.get_whitenoise(sample_rate=sample_rate)
        self.assert_batch_consistencies(F.compute_kaldi_pitch, waveform, sample_rate=sample_rate)
