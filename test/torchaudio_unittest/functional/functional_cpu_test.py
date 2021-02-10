import math
import warnings

import torch
import torchaudio
import torchaudio.functional as F
from parameterized import parameterized
import itertools

from torchaudio_unittest import common_utils
from .functional_impl import Lfilter, Spectrogram


class TestLFilterFloat32(Lfilter, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestLFilterFloat64(Lfilter, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestSpectrogramFloat32(Spectrogram, common_utils.PytorchTestCase):
    dtype = torch.float32
    device = torch.device('cpu')


class TestSpectrogramFloat64(Spectrogram, common_utils.PytorchTestCase):
    dtype = torch.float64
    device = torch.device('cpu')


class TestCreateFBMatrix(common_utils.TorchaudioTestCase):
    def test_no_warning_high_n_freq(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.create_fb_matrix(288, 0, 8000, 128, 16000)
        assert len(w) == 0

    def test_no_warning_low_n_mels(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.create_fb_matrix(201, 0, 8000, 89, 16000)
        assert len(w) == 0

    def test_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            F.create_fb_matrix(201, 0, 8000, 128, 16000)
        assert len(w) == 1


class TestComputeDeltas(common_utils.TorchaudioTestCase):
    """Test suite for correctness of compute_deltas"""
    def test_one_channel(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5]]])
        computed = F.compute_deltas(specgram, win_length=3)
        self.assertEqual(computed, expected)

    def test_two_channels(self):
        specgram = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                                  [1.0, 2.0, 3.0, 4.0]]])
        expected = torch.tensor([[[0.5, 1.0, 1.0, 0.5],
                                  [0.5, 1.0, 1.0, 0.5]]])
        computed = F.compute_deltas(specgram, win_length=3)
        self.assertEqual(computed, expected)


class TestDetectPitchFrequency(common_utils.TorchaudioTestCase):
    @parameterized.expand([(100,), (440,)])
    def test_pitch(self, frequency):
        sample_rate = 44100
        test_sine_waveform = common_utils.get_sinusoid(
            frequency=frequency, sample_rate=sample_rate, duration=5,
        )

        freq = torchaudio.functional.detect_pitch_frequency(test_sine_waveform, sample_rate)

        threshold = 1
        s = ((freq - frequency).abs() > threshold).sum()
        self.assertFalse(s)


class Testamplitude_to_DB(common_utils.TorchaudioTestCase):
    @parameterized.expand([
        ([100, 100],),
        ([2, 100, 100],),
        ([2, 2, 100, 100],),
    ])
    def test_reversible(self, shape):
        """Round trip between amplitude and db should return the original for various shape

        This implicitly also tests `DB_to_amplitude`.

        """
        amplitude_mult = 20.
        power_mult = 10.
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))

        torch.manual_seed(0)
        spec = torch.rand(*shape) * 200

        # Spectrogram amplitude -> DB -> amplitude
        db = F.amplitude_to_DB(spec, amplitude_mult, amin, db_mult, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, 0.5)

        self.assertEqual(x2, spec, atol=5e-5, rtol=1e-5)

        # Spectrogram power -> DB -> power
        db = F.amplitude_to_DB(spec, power_mult, amin, db_mult, top_db=None)
        x2 = F.DB_to_amplitude(db, ref, 1.)

        self.assertEqual(x2, spec)

    @parameterized.expand([
        ([100, 100],),
        ([2, 100, 100],),
        ([2, 2, 100, 100],),
    ])
    def test_top_db_clamp(self, shape):
        """Ensure values are properly clamped when `top_db` is supplied."""
        amplitude_mult = 20.
        amin = 1e-10
        ref = 1.0
        db_mult = math.log10(max(amin, ref))
        top_db = 40.

        torch.manual_seed(0)
        # A random tensor is used for increased entropy, but the max and min for
        # each spectrogram still need to be predictable. The max determines the
        # decibel cutoff, and the distance from the min must be large enough
        # that it triggers a clamp.
        spec = torch.rand(*shape)
        # Ensure each spectrogram has a min of 0 and a max of 1.
        spec -= spec.amin([-2, -1])[..., None, None]
        spec /= spec.amax([-2, -1])[..., None, None]
        # Expand the range to (0, 200) - wide enough to properly test clamping.
        spec *= 200

        decibels = F.amplitude_to_DB(spec, amplitude_mult, amin,
                                     db_mult, top_db=top_db)
        # Ensure the clamp was applied
        below_limit = decibels < 6.0205
        assert not below_limit.any(), (
            "{} decibel values were below the expected cutoff:\n{}".format(
                below_limit.sum().item(), decibels
            )
        )
        # Ensure it didn't over-clamp
        close_to_limit = decibels < 6.0207
        assert close_to_limit.any(), (
            f"No values were close to the limit. Did it over-clamp?\n{decibels}"
        )


class TestComplexNorm(common_utils.TorchaudioTestCase):
    @parameterized.expand(list(itertools.product(
        [(1, 2, 1025, 400, 2), (1025, 400, 2)],
        [1, 2, 0.7]
    )))
    def test_complex_norm(self, shape, power):
        torch.random.manual_seed(42)
        complex_tensor = torch.randn(*shape)
        expected_norm_tensor = complex_tensor.pow(2).sum(-1).pow(power / 2)
        norm_tensor = F.complex_norm(complex_tensor, power)
        self.assertEqual(norm_tensor, expected_norm_tensor, atol=1e-5, rtol=1e-5)


class TestMaskAlongAxis(common_utils.TorchaudioTestCase):
    @parameterized.expand(list(itertools.product(
        [(2, 1025, 400), (1, 201, 100)],
        [100],
        [0., 30.],
        [1, 2]
    )))
    def test_mask_along_axis(self, shape, mask_param, mask_value, axis):
        torch.random.manual_seed(42)
        specgram = torch.randn(*shape)
        mask_specgram = F.mask_along_axis(specgram, mask_param, mask_value, axis)

        other_axis = 1 if axis == 2 else 2

        masked_columns = (mask_specgram == mask_value).sum(other_axis)
        num_masked_columns = (masked_columns == mask_specgram.size(other_axis)).sum()
        num_masked_columns //= mask_specgram.size(0)

        assert mask_specgram.size() == specgram.size()
        assert num_masked_columns < mask_param


class TestMaskAlongAxisIID(common_utils.TorchaudioTestCase):
    @parameterized.expand(list(itertools.product(
        [100],
        [0., 30.],
        [2, 3]
    )))
    def test_mask_along_axis_iid(self, mask_param, mask_value, axis):
        torch.random.manual_seed(42)
        specgrams = torch.randn(4, 2, 1025, 400)

        mask_specgrams = F.mask_along_axis_iid(specgrams, mask_param, mask_value, axis)

        other_axis = 2 if axis == 3 else 3

        masked_columns = (mask_specgrams == mask_value).sum(other_axis)
        num_masked_columns = (masked_columns == mask_specgrams.size(other_axis)).sum(-1)

        assert mask_specgrams.size() == specgrams.size()
        assert (num_masked_columns < mask_param).sum() == num_masked_columns.numel()
