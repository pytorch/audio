import io
import unittest
from itertools import product

from torchaudio.backend import sox_io_backend
from parameterized import parameterized

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    TorchaudioTestCase,
    PytorchTestCase,
    skipIfNoExec,
    skipIfNoSox,
    get_wav_data,
    load_wav,
    save_wav,
    sox_utils,
)
from .common import (
    name_func,
    get_enc_params,
)


def _get_sox_encoding(encoding):
    encodings = {
        'PCM_F': 'floating-point',
        'PCM_S': 'signed-integer',
        'PCM_U': 'unsigned-integer',
        'ULAW': 'u-law',
        'ALAW': 'a-law',
    }
    return encodings.get(encoding)


class SaveTestBase(TempDirMixin, TorchaudioTestCase):
    def assert_save_consistency(
            self,
            format: str,
            *,
            compression: float = None,
            encoding: str = None,
            bits_per_sample: int = None,
            sample_rate: float = 8000,
            num_channels: int = 2,
            num_frames: float = 3 * 8000,
            src_dtype: str = 'int32',
            test_mode: str = "path",
    ):
        """`save` function produces file that is comparable with `sox` command

        To compare that the file produced by `save` function agains the file produced by
        the equivalent `sox` command, we need to load both files.
        But there are many formats that cannot be opened with common Python modules (like
        SciPy).
        So we use `sox` command to prepare the original data and convert the saved files
        into a format that SciPy can read (PCM wav).
        The following diagram illustrates this process. The difference is 2.1. and 3.1.

        This assumes that
         - loading data with SciPy preserves the data well.
         - converting the resulting files into WAV format with `sox` preserve the data well.

                          x
                          | 1. Generate source wav file with SciPy
                          |
                          v
          -------------- wav ----------------
         |                                   |
         | 2.1. load with scipy              | 3.1. Convert to the target
         |   then save it into the target    |      format depth with sox
         |   format with torchaudio          |
         v                                   v
        target format                       target format
         |                                   |
         | 2.2. Convert to wav with sox      | 3.2. Convert to wav with sox
         |                                   |
         v                                   v
        wav                                 wav
         |                                   |
         | 2.3. load with scipy              | 3.3. load with scipy
         |                                   |
         v                                   v
        tensor -------> compare <--------- tensor

        """
        cmp_encoding = 'floating-point'
        cmp_bit_depth = 32

        src_path = self.get_temp_path('1.source.wav')
        tgt_path = self.get_temp_path(f'2.1.torchaudio.{format}')
        tst_path = self.get_temp_path('2.2.result.wav')
        sox_path = self.get_temp_path(f'3.1.sox.{format}')
        ref_path = self.get_temp_path('3.2.ref.wav')

        # 1. Generate original wav
        data = get_wav_data(src_dtype, num_channels, normalize=False, num_frames=num_frames)
        save_wav(src_path, data, sample_rate)

        # 2.1. Convert the original wav to target format with torchaudio
        data = load_wav(src_path, normalize=False)[0]
        if test_mode == "path":
            sox_io_backend.save(
                tgt_path, data, sample_rate,
                compression=compression, encoding=encoding, bits_per_sample=bits_per_sample)
        elif test_mode == "fileobj":
            with open(tgt_path, 'bw') as file_:
                sox_io_backend.save(
                    file_, data, sample_rate,
                    format=format, compression=compression,
                    encoding=encoding, bits_per_sample=bits_per_sample)
        elif test_mode == "bytesio":
            file_ = io.BytesIO()
            sox_io_backend.save(
                file_, data, sample_rate,
                format=format, compression=compression,
                encoding=encoding, bits_per_sample=bits_per_sample)
            file_.seek(0)
            with open(tgt_path, 'bw') as f:
                f.write(file_.read())
        else:
            raise ValueError(f"Unexpected test mode: {test_mode}")
        # 2.2. Convert the target format to wav with sox
        sox_utils.convert_audio_file(
            tgt_path, tst_path, encoding=cmp_encoding, bit_depth=cmp_bit_depth)
        # 2.3. Load with SciPy
        found = load_wav(tst_path, normalize=False)[0]

        # 3.1. Convert the original wav to target format with sox
        sox_encoding = _get_sox_encoding(encoding)
        sox_utils.convert_audio_file(
            src_path, sox_path,
            compression=compression, encoding=sox_encoding, bit_depth=bits_per_sample)
        # 3.2. Convert the target format to wav with sox
        sox_utils.convert_audio_file(
            sox_path, ref_path, encoding=cmp_encoding, bit_depth=cmp_bit_depth)
        # 3.3. Load with SciPy
        expected = load_wav(ref_path, normalize=False)[0]

        self.assertEqual(found, expected)


def nested_params(*params):
    def _name_func(func, _, params):
        strs = []
        for arg in params.args:
            if isinstance(arg, tuple):
                strs.append("_".join(str(a) for a in arg))
            else:
                strs.append(str(arg))
        return f'{func.__name__}_{"_".join(strs)}'

    return parameterized.expand(
        list(product(*params)),
        name_func=_name_func
    )


@skipIfNoExec('sox')
@skipIfNoSox
class SaveTest(SaveTestBase):
    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            ('PCM_U', 8),
            ('PCM_S', 16),
            ('PCM_S', 32),
            ('PCM_F', 32),
            ('PCM_F', 64),
            ('ULAW', 8),
            ('ALAW', 8),
        ],
    )
    def test_save_wav(self, test_mode, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency(
            "wav", encoding=encoding, bits_per_sample=bits_per_sample, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            ('float32', ),
            ('int32', ),
            ('int16', ),
            ('uint8', ),
        ],
    )
    def test_save_wav_dtype(self, test_mode, params):
        dtype, = params
        self.assert_save_consistency(
            "wav", src_dtype=dtype, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            None,
            -4.2,
            -0.2,
            0,
            0.2,
            96,
            128,
            160,
            192,
            224,
            256,
            320,
        ],
    )
    def test_save_mp3(self, test_mode, bit_rate):
        if test_mode in ["fileobj", "bytesio"]:
            if bit_rate is not None and bit_rate < 1:
                raise unittest.SkipTest(
                    "mp3 format with variable bit rate is known to "
                    "not yield the exact same result as sox command.")
        self.assert_save_consistency(
            "mp3", compression=bit_rate, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [8, 16, 24],
        [
            None,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ],
    )
    def test_save_flac(self, test_mode, bits_per_sample, compression_level):
        self.assert_save_consistency(
            "flac", compression=compression_level,
            bits_per_sample=bits_per_sample, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
    )
    def test_save_htk(self, test_mode):
        self.assert_save_consistency("htk", test_mode=test_mode, num_channels=1)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            None,
            -1,
            0,
            1,
            2,
            3,
            3.6,
            5,
            10,
        ],
    )
    def test_save_vorbis(self, test_mode, quality_level):
        self.assert_save_consistency(
            "vorbis", compression=quality_level, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            ('PCM_S', 8, ),
            ('PCM_S', 16, ),
            ('PCM_S', 24, ),
            ('PCM_S', 32, ),
            ('ULAW', 8),
            ('ALAW', 8),
            ('ALAW', 16),
            ('ALAW', 24),
            ('ALAW', 32),
        ],
    )
    def test_save_sphere(self, test_mode, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency(
            "sph", encoding=encoding, bits_per_sample=bits_per_sample, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            ('PCM_U', 8, ),
            ('PCM_S', 16, ),
            ('PCM_S', 24, ),
            ('PCM_S', 32, ),
            ('PCM_F', 32, ),
            ('PCM_F', 64, ),
            ('ULAW', 8, ),
            ('ALAW', 8, ),
        ],
    )
    def test_save_amb(self, test_mode, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency(
            "amb", encoding=encoding, bits_per_sample=bits_per_sample, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            None,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
    )
    def test_save_amr_nb(self, test_mode, bit_rate):
        self.assert_save_consistency(
            "amr-nb", compression=bit_rate, num_channels=1, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
    )
    def test_save_gsm(self, test_mode):
        self.assert_save_consistency(
            "gsm", test_mode=test_mode)

    @parameterized.expand([
        ("wav", "PCM_S", 16),
        ("mp3", ),
        ("flac", ),
        ("vorbis", ),
        ("sph", "PCM_S", 16),
        ("amr-nb", ),
        ("amb", "PCM_S", 16),
    ], name_func=name_func)
    def test_save_large(self, format, encoding=None, bits_per_sample=None):
        """`sox_io_backend.save` can save large files."""
        sample_rate = 8000
        one_hour = 60 * 60 * sample_rate
        self.assert_save_consistency(
            format, num_channels=1, sample_rate=8000, num_frames=one_hour,
            encoding=encoding, bits_per_sample=bits_per_sample)

    @parameterized.expand([
        (32, ),
        (64, ),
        (128, ),
        (256, ),
    ], name_func=name_func)
    def test_save_multi_channels(self, num_channels):
        """`sox_io_backend.save` can save audio with many channels"""
        self.assert_save_consistency(
            "wav", encoding="PCM_S", bits_per_sample=16,
            num_channels=num_channels)


@skipIfNoExec('sox')
@skipIfNoSox
class TestSaveParams(TempDirMixin, PytorchTestCase):
    """Test the correctness of optional parameters of `sox_io_backend.save`"""
    @parameterized.expand([(True, ), (False, )], name_func=name_func)
    def test_save_channels_first(self, channels_first):
        """channels_first swaps axes"""
        path = self.get_temp_path('data.wav')
        data = get_wav_data(
            'int16', 2, channels_first=channels_first, normalize=False)
        sox_io_backend.save(
            path, data, 8000, channels_first=channels_first)
        found = load_wav(path, normalize=False)[0]
        expected = data if channels_first else data.transpose(1, 0)
        self.assertEqual(found, expected)

    @parameterized.expand([
        'float32', 'int32', 'int16', 'uint8'
    ], name_func=name_func)
    def test_save_noncontiguous(self, dtype):
        """Noncontiguous tensors are saved correctly"""
        path = self.get_temp_path('data.wav')
        enc, bps = get_enc_params(dtype)
        expected = get_wav_data(dtype, 4, normalize=False)[::2, ::2]
        assert not expected.is_contiguous()
        sox_io_backend.save(
            path, expected, 8000, encoding=enc, bits_per_sample=bps)
        found = load_wav(path, normalize=False)[0]
        self.assertEqual(found, expected)

    @parameterized.expand([
        'float32', 'int32', 'int16', 'uint8',
    ])
    def test_save_tensor_preserve(self, dtype):
        """save function should not alter Tensor"""
        path = self.get_temp_path('data.wav')
        expected = get_wav_data(dtype, 4, normalize=False)[::2, ::2]

        data = expected.clone()
        sox_io_backend.save(path, data, 8000)

        self.assertEqual(data, expected)
