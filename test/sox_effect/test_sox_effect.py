import itertools

from torchaudio import sox_effects
from parameterized import parameterized

from ..common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoExtension,
    get_sinusoid,
    get_wav_data,
    save_wav,
    load_wav,
    load_params,
    sox_utils,
)
from .common import (
    name_func,
)


@skipIfNoExtension
class TestSoxEffects(PytorchTestCase):
    def test_init(self):
        """Calling init_sox_effects multiple times does not crush"""
        for _ in range(3):
            sox_effects.init_sox_effects()


@skipIfNoExtension
class TestSoxEffectsTensor(TempDirMixin, PytorchTestCase):
    """Test suite for `apply_effects_tensor` function"""
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2, 4, 8],
        [True, False]
    )), name_func=name_func)
    def test_apply_no_effect(self, dtype, sample_rate, num_channels, channels_first):
        """`apply_effects_tensor` without effects should return identical data as input"""
        original = get_wav_data(dtype, num_channels, channels_first=channels_first)
        expected = original.clone()
        found, output_sample_rate = sox_effects.apply_effects_tensor(
            expected, sample_rate, [], channels_first)

        assert output_sample_rate == sample_rate
        # SoxEffect should not alter the input Tensor object
        self.assertEqual(original, expected)
        # SoxEffect should not return the same Tensor object
        assert expected is not found
        # Returned Tensor should equal to the input Tensor
        self.assertEqual(expected, found)

    @parameterized.expand(
        load_params("sox_effect_test_args.json"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects(self, args):
        """`apply_effects_tensor` should return identical data as sox command"""
        effects = args['effects']
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)
        output_sr = args.get("output_sample_rate")

        input_path = self.get_temp_path('input.wav')
        reference_path = self.get_temp_path('reference.wav')

        original = get_sinusoid(
            frequency=800, sample_rate=input_sr,
            n_channels=num_channels, dtype='float32')
        save_wav(input_path, original, input_sr)
        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_sample_rate=output_sr)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_tensor(original, input_sr, effects)

        assert sr == expected_sr
        self.assertEqual(expected, found)


@skipIfNoExtension
class TestSoxEffectsFile(TempDirMixin, PytorchTestCase):
    """Test suite for `apply_effects_file` function"""
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2, 4, 8],
        [False, True],
    )), name_func=name_func)
    def test_apply_no_effect(self, dtype, sample_rate, num_channels, channels_first):
        """`apply_effects_file` without effects should return identical data as input"""
        path = self.get_temp_path('input.wav')
        expected = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(path, expected, sample_rate, channels_first=channels_first)

        found, output_sample_rate = sox_effects.apply_effects_file(
            path, [], normalize=False, channels_first=channels_first)

        assert output_sample_rate == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand(
        load_params("sox_effect_test_args.json"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects(self, args):
        """`apply_effects_file` should return identical data as sox command"""
        dtype = 'int32'
        channels_first = True
        effects = args['effects']
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)
        output_sr = args.get("output_sample_rate")

        input_path = self.get_temp_path('input.wav')
        reference_path = self.get_temp_path('reference.wav')
        data = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(input_path, data, input_sr, channels_first=channels_first)
        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_sample_rate=output_sr)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            input_path, effects, normalize=False, channels_first=channels_first)

        assert sr == expected_sr
        self.assertEqual(found, expected)


@skipIfNoExtension
class TestFileFormats(TempDirMixin, PytorchTestCase):
    """`apply_effects_file` gives the same result as sox on various file formats"""
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}')
    def test_wav(self, dtype, sample_rate, num_channels):
        """`apply_effects_file` works on various wav format"""
        channels_first = True
        effects = [['band', '300', '10']]

        input_path = self.get_temp_path('input.wav')
        reference_path = self.get_temp_path('reference.wav')
        data = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(input_path, data, sample_rate, channels_first=channels_first)
        sox_utils.run_sox_effect(input_path, reference_path, effects)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            input_path, effects, normalize=False, channels_first=channels_first)

        assert sr == expected_sr
        self.assertEqual(found, expected)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
    )), name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}')
    def test_mp3(self, sample_rate, num_channels):
        """`apply_effects_file` works on various mp3 format"""
        channels_first = True
        effects = [['band', '300', '10']]

        input_path = self.get_temp_path('input.mp3')
        reference_path = self.get_temp_path('reference.wav')
        sox_utils.gen_audio_file(input_path, sample_rate, num_channels)
        sox_utils.run_sox_effect(input_path, reference_path, effects)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            input_path, effects, channels_first=channels_first)
        save_wav(self.get_temp_path('result.wav'), found, sr, channels_first=channels_first)

        assert sr == expected_sr
        self.assertEqual(found, expected, atol=1e-4, rtol=1e-8)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
    )), name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}')
    def test_flac(self, sample_rate, num_channels):
        """`apply_effects_file` works on various flac format"""
        channels_first = True
        effects = [['band', '300', '10']]

        input_path = self.get_temp_path('input.flac')
        reference_path = self.get_temp_path('reference.wav')
        sox_utils.gen_audio_file(input_path, sample_rate, num_channels)
        sox_utils.run_sox_effect(input_path, reference_path, effects, output_bitdepth=32)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            input_path, effects, channels_first=channels_first)
        save_wav(self.get_temp_path('result.wav'), found, sr, channels_first=channels_first)

        assert sr == expected_sr
        self.assertEqual(found, expected)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
    )), name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}')
    def test_vorbis(self, sample_rate, num_channels):
        """`apply_effects_file` works on various vorbis format"""
        channels_first = True
        effects = [['band', '300', '10']]

        input_path = self.get_temp_path('input.vorbis')
        reference_path = self.get_temp_path('reference.wav')
        sox_utils.gen_audio_file(input_path, sample_rate, num_channels)
        sox_utils.run_sox_effect(input_path, reference_path, effects, output_bitdepth=32)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            input_path, effects, channels_first=channels_first)
        save_wav(self.get_temp_path('result.wav'), found, sr, channels_first=channels_first)

        assert sr == expected_sr
        self.assertEqual(found, expected)
