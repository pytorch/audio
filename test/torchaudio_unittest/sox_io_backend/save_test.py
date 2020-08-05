import itertools

from torchaudio.backend import sox_io_backend
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    PytorchTestCase,
    skipIfNoExec,
    skipIfNoExtension,
    get_wav_data,
    load_wav,
    save_wav,
    sox_utils,
)
from .common import (
    name_func,
)


class SaveTestBase(TempDirMixin, PytorchTestCase):
    def assert_wav(self, dtype, sample_rate, num_channels, num_frames):
        """`sox_io_backend.save` can save wav format."""
        path = self.get_temp_path('data.wav')
        expected = get_wav_data(dtype, num_channels, num_frames=num_frames)
        sox_io_backend.save(path, expected, sample_rate)
        found, sr = load_wav(path)
        assert sample_rate == sr
        self.assertEqual(found, expected)

    def assert_mp3(self, sample_rate, num_channels, bit_rate, duration):
        """`sox_io_backend.save` can save mp3 format.

        mp3 encoding introduces delay and boundary effects so
        we convert the resulting mp3 to wav and compare the results there

                          |
                          | 1. Generate original wav file with SciPy
                          |
                          v
          -------------- wav ----------------
         |                                   |
         | 2.1. load with scipy              | 3.1. Convert to mp3 with Sox
         | then save with torchaudio         |
         v                                   v
        mp3                                 mp3
         |                                   |
         | 2.2. Convert to wav with Sox      | 3.2. Convert to wav with Sox
         |                                   |
         v                                   v
        wav                                 wav
         |                                   |
         | 2.3. load with scipy              | 3.3. load with scipy
         |                                   |
         v                                   v
        tensor -------> compare <--------- tensor

        """
        src_path = self.get_temp_path('1.reference.wav')
        mp3_path = self.get_temp_path('2.1.torchaudio.mp3')
        wav_path = self.get_temp_path('2.2.torchaudio.wav')
        mp3_path_sox = self.get_temp_path('3.1.sox.mp3')
        wav_path_sox = self.get_temp_path('3.2.sox.wav')

        # 1. Generate original wav
        data = get_wav_data('float32', num_channels, normalize=True, num_frames=duration * sample_rate)
        save_wav(src_path, data, sample_rate)
        # 2.1. Convert the original wav to mp3 with torchaudio
        sox_io_backend.save(
            mp3_path, load_wav(src_path)[0], sample_rate, compression=bit_rate)
        # 2.2. Convert the mp3 to wav with Sox
        sox_utils.convert_audio_file(mp3_path, wav_path)
        # 2.3. Load
        found = load_wav(wav_path)[0]

        # 3.1. Convert the original wav to mp3 with SoX
        sox_utils.convert_audio_file(src_path, mp3_path_sox, compression=bit_rate)
        # 3.2. Convert the mp3 to wav with Sox
        sox_utils.convert_audio_file(mp3_path_sox, wav_path_sox)
        # 3.3. Load
        expected = load_wav(wav_path_sox)[0]

        self.assertEqual(found, expected)

    def assert_flac(self, sample_rate, num_channels, compression_level, duration):
        """`sox_io_backend.save` can save flac format.

        This test takes the same strategy as mp3 to compare the result
        """
        src_path = self.get_temp_path('1.reference.wav')
        flc_path = self.get_temp_path('2.1.torchaudio.flac')
        wav_path = self.get_temp_path('2.2.torchaudio.wav')
        flc_path_sox = self.get_temp_path('3.1.sox.flac')
        wav_path_sox = self.get_temp_path('3.2.sox.wav')

        # 1. Generate original wav
        data = get_wav_data('float32', num_channels, normalize=True, num_frames=duration * sample_rate)
        save_wav(src_path, data, sample_rate)
        # 2.1. Convert the original wav to flac with torchaudio
        sox_io_backend.save(
            flc_path, load_wav(src_path)[0], sample_rate, compression=compression_level)
        # 2.2. Convert the flac to wav with Sox
        # converting to 32 bit because flac file has 24 bit depth which scipy cannot handle.
        sox_utils.convert_audio_file(flc_path, wav_path, bit_depth=32)
        # 2.3. Load
        found = load_wav(wav_path)[0]

        # 3.1. Convert the original wav to flac with SoX
        sox_utils.convert_audio_file(src_path, flc_path_sox, compression=compression_level)
        # 3.2. Convert the flac to wav with Sox
        # converting to 32 bit because flac file has 24 bit depth which scipy cannot handle.
        sox_utils.convert_audio_file(flc_path_sox, wav_path_sox, bit_depth=32)
        # 3.3. Load
        expected = load_wav(wav_path_sox)[0]

        self.assertEqual(found, expected)

    def _assert_vorbis(self, sample_rate, num_channels, quality_level, duration):
        """`sox_io_backend.save` can save vorbis format.

        This test takes the same strategy as mp3 to compare the result
        """
        src_path = self.get_temp_path('1.reference.wav')
        vbs_path = self.get_temp_path('2.1.torchaudio.vorbis')
        wav_path = self.get_temp_path('2.2.torchaudio.wav')
        vbs_path_sox = self.get_temp_path('3.1.sox.vorbis')
        wav_path_sox = self.get_temp_path('3.2.sox.wav')

        # 1. Generate original wav
        data = get_wav_data('int16', num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(src_path, data, sample_rate)
        # 2.1. Convert the original wav to vorbis with torchaudio
        sox_io_backend.save(
            vbs_path, load_wav(src_path)[0], sample_rate, compression=quality_level)
        # 2.2. Convert the vorbis to wav with Sox
        sox_utils.convert_audio_file(vbs_path, wav_path)
        # 2.3. Load
        found = load_wav(wav_path)[0]

        # 3.1. Convert the original wav to vorbis with SoX
        sox_utils.convert_audio_file(src_path, vbs_path_sox, compression=quality_level)
        # 3.2. Convert the vorbis to wav with Sox
        sox_utils.convert_audio_file(vbs_path_sox, wav_path_sox)
        # 3.3. Load
        expected = load_wav(wav_path_sox)[0]

        # sox's vorbis encoding has some random boundary effect, which cause small number of
        # samples yields higher descrepency than the others.
        # so we allow small portions of data to be outside of absolute torelance.
        # make sure to pass somewhat long duration
        atol = 1.0e-4
        max_failure_allowed = 0.01  # this percent of samples are allowed to outside of atol.
        failure_ratio = ((found - expected).abs() > atol).sum().item() / found.numel()
        if failure_ratio > max_failure_allowed:
            # it's failed and this will give a better error message.
            self.assertEqual(found, expected, atol=atol, rtol=1.3e-6)

    def assert_vorbis(self, *args, **kwargs):
        # sox's vorbis encoding has some randomness, so we run tests multiple time
        max_retry = 5
        error = None
        for _ in range(max_retry):
            try:
                self._assert_vorbis(*args, **kwargs)
                break
            except AssertionError as e:
                error = e
        else:
            raise error


@skipIfNoExec('sox')
@skipIfNoExtension
class TestSave(SaveTestBase):
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
    )), name_func=name_func)
    def test_wav(self, dtype, sample_rate, num_channels):
        """`sox_io_backend.save` can save wav format."""
        self.assert_wav(dtype, sample_rate, num_channels, num_frames=None)

    @parameterized.expand(list(itertools.product(
        ['float32'],
        [16000],
        [2],
    )), name_func=name_func)
    def test_wav_large(self, dtype, sample_rate, num_channels):
        """`sox_io_backend.save` can save large wav file."""
        two_hours = 2 * 60 * 60 * sample_rate
        self.assert_wav(dtype, sample_rate, num_channels, num_frames=two_hours)

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [4, 8, 16, 32],
    )), name_func=name_func)
    def test_multiple_channels(self, dtype, num_channels):
        """`sox_io_backend.save` can save wav with more than 2 channels."""
        sample_rate = 8000
        self.assert_wav(dtype, sample_rate, num_channels, num_frames=None)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [-4.2, -0.2, 0, 0.2, 96, 128, 160, 192, 224, 256, 320],
    )), name_func=name_func)
    def test_mp3(self, sample_rate, num_channels, bit_rate):
        """`sox_io_backend.save` can save mp3 format."""
        self.assert_mp3(sample_rate, num_channels, bit_rate, duration=1)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [128],
    )), name_func=name_func)
    def test_mp3_large(self, sample_rate, num_channels, bit_rate):
        """`sox_io_backend.save` can save large mp3 file."""
        two_hours = 2 * 60 * 60
        self.assert_mp3(sample_rate, num_channels, bit_rate, duration=two_hours)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        list(range(9)),
    )), name_func=name_func)
    def test_flac(self, sample_rate, num_channels, compression_level):
        """`sox_io_backend.save` can save flac format."""
        self.assert_flac(sample_rate, num_channels, compression_level, duration=1)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [0],
    )), name_func=name_func)
    def test_flac_large(self, sample_rate, num_channels, compression_level):
        """`sox_io_backend.save` can save large flac file."""
        two_hours = 2 * 60 * 60
        self.assert_flac(sample_rate, num_channels, compression_level, duration=two_hours)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [-1, 0, 1, 2, 3, 3.6, 5, 10],
    )), name_func=name_func)
    def test_vorbis(self, sample_rate, num_channels, quality_level):
        """`sox_io_backend.save` can save vorbis format."""
        self.assert_vorbis(sample_rate, num_channels, quality_level, duration=20)

    # note: torchaudio can load large vorbis file, but cannot save large volbis file
    # the following test causes Segmentation fault
    #
    '''
    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [10],
    )), name_func=name_func)
    def test_vorbis_large(self, sample_rate, num_channels, quality_level):
        """`sox_io_backend.save` can save large vorbis file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_vorbis(sample_rate, num_channels, quality_level, two_hours)
    '''


@skipIfNoExec('sox')
@skipIfNoExtension
class TestSaveParams(TempDirMixin, PytorchTestCase):
    """Test the correctness of optional parameters of `sox_io_backend.save`"""
    @parameterized.expand([(True, ), (False, )], name_func=name_func)
    def test_channels_first(self, channels_first):
        """channels_first swaps axes"""
        path = self.get_temp_path('data.wav')
        data = get_wav_data('int32', 2, channels_first=channels_first)
        sox_io_backend.save(
            path, data, 8000, channels_first=channels_first)
        found = load_wav(path)[0]
        expected = data if channels_first else data.transpose(1, 0)
        self.assertEqual(found, expected)

    @parameterized.expand([
        'float32', 'int32', 'int16', 'uint8'
    ], name_func=name_func)
    def test_noncontiguous(self, dtype):
        """Noncontiguous tensors are saved correctly"""
        path = self.get_temp_path('data.wav')
        expected = get_wav_data(dtype, 4)[::2, ::2]
        assert not expected.is_contiguous()
        sox_io_backend.save(path, expected, 8000)
        found = load_wav(path)[0]
        self.assertEqual(found, expected)

    @parameterized.expand([
        'float32', 'int32', 'int16', 'uint8',
    ])
    def test_tensor_preserve(self, dtype):
        """save function should not alter Tensor"""
        path = self.get_temp_path('data.wav')
        expected = get_wav_data(dtype, 4)[::2, ::2]

        data = expected.clone()
        sox_io_backend.save(path, data, 8000)

        self.assertEqual(data, expected)
