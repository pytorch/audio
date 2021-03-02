from torchaudio import sox_effects
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    skipIfNoSox,
    get_wav_data,
    get_sinusoid,
    save_wav,
)
from .common import (
    load_params,
)


@skipIfNoSox
class SmokeTest(TempDirMixin, TorchaudioTestCase):
    """Run smoke test on various effects

    The purpose of this test suite is to verify that sox_effect functionalities do not exhibit
    abnormal behaviors.

    This test suite should be able to run without any additional tools (such as sox command),
    however without such tools, the correctness of each function cannot be verified.
    """
    @parameterized.expand(
        load_params("sox_effect_test_args.json"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects_tensor(self, args):
        """`apply_effects_tensor` should not crash"""
        effects = args['effects']
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)
        original = get_sinusoid(
            frequency=800, sample_rate=input_sr,
            n_channels=num_channels, dtype='float32')
        _found, _sr = sox_effects.apply_effects_tensor(original, input_sr, effects)

    @parameterized.expand(
        load_params("sox_effect_test_args.json"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects_file(self, args):
        """`apply_effects_file` should return identical data as sox command"""
        dtype = 'int32'
        channels_first = True
        effects = args['effects']
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)

        input_path = self.get_temp_path('input.wav')
        data = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(input_path, data, input_sr, channels_first=channels_first)

        _found, _sr = sox_effects.apply_effects_file(
            input_path, effects, normalize=False, channels_first=channels_first)

    @parameterized.expand(
        load_params("sox_effect_test_args.json"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects_fileobj(self, args):
        """`apply_effects_file` should return identical data as sox command"""
        dtype = 'int32'
        channels_first = True
        effects = args['effects']
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)

        input_path = self.get_temp_path('input.wav')
        data = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(input_path, data, input_sr, channels_first=channels_first)

        with open(input_path, 'rb') as fileobj:
            _found, _sr = sox_effects.apply_effects_file(
                fileobj, effects, normalize=False, channels_first=channels_first)
