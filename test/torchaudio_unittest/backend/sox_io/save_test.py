import os

import torch
from parameterized import parameterized
from torchaudio.backend import sox_io_backend
from torchaudio_unittest.common_utils import (
    get_wav_data,
    load_wav,
    nested_params,
    PytorchTestCase,
    save_wav,
    skipIfNoExec,
    skipIfNoSox,
    skipIfNoSoxEncoder,
    sox_utils,
    TempDirMixin,
    TorchaudioTestCase,
)

from .common import get_enc_params, name_func


def _get_sox_encoding(encoding):
    encodings = {
        "PCM_F": "floating-point",
        "PCM_S": "signed-integer",
        "PCM_U": "unsigned-integer",
        "ULAW": "u-law",
        "ALAW": "a-law",
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
        src_dtype: str = "int32",
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
        cmp_encoding = "floating-point"
        cmp_bit_depth = 32

        src_path = self.get_temp_path("1.source.wav")
        tgt_path = self.get_temp_path(f"2.1.torchaudio.{format}")
        tst_path = self.get_temp_path("2.2.result.wav")
        sox_path = self.get_temp_path(f"3.1.sox.{format}")
        ref_path = self.get_temp_path("3.2.ref.wav")

        # 1. Generate original wav
        data = get_wav_data(src_dtype, num_channels, normalize=False, num_frames=num_frames)
        save_wav(src_path, data, sample_rate)

        # 2.1. Convert the original wav to target format with torchaudio
        data = load_wav(src_path, normalize=False)[0]
        sox_io_backend.save(
            tgt_path, data, sample_rate, compression=compression, encoding=encoding, bits_per_sample=bits_per_sample
        )
        # 2.2. Convert the target format to wav with sox
        sox_utils.convert_audio_file(tgt_path, tst_path, encoding=cmp_encoding, bit_depth=cmp_bit_depth)
        # 2.3. Load with SciPy
        found = load_wav(tst_path, normalize=False)[0]

        # 3.1. Convert the original wav to target format with sox
        sox_encoding = _get_sox_encoding(encoding)
        sox_utils.convert_audio_file(
            src_path, sox_path, compression=compression, encoding=sox_encoding, bit_depth=bits_per_sample
        )
        # 3.2. Convert the target format to wav with sox
        sox_utils.convert_audio_file(sox_path, ref_path, encoding=cmp_encoding, bit_depth=cmp_bit_depth)
        # 3.3. Load with SciPy
        expected = load_wav(ref_path, normalize=False)[0]

        self.assertEqual(found, expected)


@skipIfNoExec("sox")
@skipIfNoSox
class SaveTest(SaveTestBase):
    @nested_params(
        [
            ("PCM_U", 8),
            ("PCM_S", 16),
            ("PCM_S", 32),
            ("PCM_F", 32),
            ("PCM_F", 64),
            ("ULAW", 8),
            ("ALAW", 8),
        ],
    )
    def test_save_wav(self, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency("wav", encoding=encoding, bits_per_sample=bits_per_sample)

    @nested_params(
        [
            ("float32",),
            ("int32",),
            ("int16",),
            ("uint8",),
        ],
    )
    def test_save_wav_dtype(self, params):
        (dtype,) = params
        self.assert_save_consistency("wav", src_dtype=dtype)

    @nested_params(
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
    def test_save_flac(self, bits_per_sample, compression_level):
        self.assert_save_consistency("flac", compression=compression_level, bits_per_sample=bits_per_sample)

    def test_save_htk(self):
        self.assert_save_consistency("htk", num_channels=1)

    @nested_params(
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
    def test_save_vorbis(self, quality_level):
        self.assert_save_consistency("vorbis", compression=quality_level)

    @nested_params(
        [
            (
                "PCM_S",
                8,
            ),
            (
                "PCM_S",
                16,
            ),
            (
                "PCM_S",
                24,
            ),
            (
                "PCM_S",
                32,
            ),
            ("ULAW", 8),
            ("ALAW", 8),
            ("ALAW", 16),
            ("ALAW", 24),
            ("ALAW", 32),
        ],
    )
    def test_save_sphere(self, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency("sph", encoding=encoding, bits_per_sample=bits_per_sample)

    @nested_params(
        [
            (
                "PCM_U",
                8,
            ),
            (
                "PCM_S",
                16,
            ),
            (
                "PCM_S",
                24,
            ),
            (
                "PCM_S",
                32,
            ),
            (
                "PCM_F",
                32,
            ),
            (
                "PCM_F",
                64,
            ),
            (
                "ULAW",
                8,
            ),
            (
                "ALAW",
                8,
            ),
        ],
    )
    def test_save_amb(self, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency("amb", encoding=encoding, bits_per_sample=bits_per_sample)

    @nested_params(
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
    @skipIfNoSoxEncoder("amr-nb")
    def test_save_amr_nb(self, bit_rate):
        self.assert_save_consistency("amr-nb", compression=bit_rate, num_channels=1)

    def test_save_gsm(self):
        self.assert_save_consistency("gsm", num_channels=1)
        with self.assertRaises(RuntimeError, msg="gsm format only supports single channel audio."):
            self.assert_save_consistency("gsm", num_channels=2)
        with self.assertRaises(RuntimeError, msg="gsm format only supports a sampling rate of 8kHz."):
            self.assert_save_consistency("gsm", sample_rate=16000)

    @parameterized.expand(
        [
            ("wav", "PCM_S", 16),
            ("flac",),
            ("vorbis",),
            ("sph", "PCM_S", 16),
            ("amb", "PCM_S", 16),
        ],
        name_func=name_func,
    )
    def test_save_large(self, format, encoding=None, bits_per_sample=None):
        self._test_save_large(format, encoding, bits_per_sample)

    @skipIfNoSoxEncoder("amr-nb")
    def test_save_large_amr_nb(self):
        self._test_save_large("amr-nb")

    def _test_save_large(self, format, encoding=None, bits_per_sample=None):
        """`sox_io_backend.save` can save large files."""
        sample_rate = 8000
        one_hour = 60 * 60 * sample_rate
        self.assert_save_consistency(
            format,
            num_channels=1,
            sample_rate=8000,
            num_frames=one_hour,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
        )

    @parameterized.expand(
        [
            (32,),
            (64,),
            (128,),
            (256,),
        ],
        name_func=name_func,
    )
    def test_save_multi_channels(self, num_channels):
        """`sox_io_backend.save` can save audio with many channels"""
        self.assert_save_consistency("wav", encoding="PCM_S", bits_per_sample=16, num_channels=num_channels)


@skipIfNoExec("sox")
@skipIfNoSox
class TestSaveParams(TempDirMixin, PytorchTestCase):
    """Test the correctness of optional parameters of `sox_io_backend.save`"""

    @parameterized.expand([(True,), (False,)], name_func=name_func)
    def test_save_channels_first(self, channels_first):
        """channels_first swaps axes"""
        path = self.get_temp_path("data.wav")
        data = get_wav_data("int16", 2, channels_first=channels_first, normalize=False)
        sox_io_backend.save(path, data, 8000, channels_first=channels_first)
        found = load_wav(path, normalize=False)[0]
        expected = data if channels_first else data.transpose(1, 0)
        self.assertEqual(found, expected)

    @parameterized.expand(["float32", "int32", "int16", "uint8"], name_func=name_func)
    def test_save_noncontiguous(self, dtype):
        """Noncontiguous tensors are saved correctly"""
        path = self.get_temp_path("data.wav")
        enc, bps = get_enc_params(dtype)
        expected = get_wav_data(dtype, 4, normalize=False)[::2, ::2]
        assert not expected.is_contiguous()
        sox_io_backend.save(path, expected, 8000, encoding=enc, bits_per_sample=bps)
        found = load_wav(path, normalize=False)[0]
        self.assertEqual(found, expected)

    @parameterized.expand(
        [
            "float32",
            "int32",
            "int16",
            "uint8",
        ]
    )
    def test_save_tensor_preserve(self, dtype):
        """save function should not alter Tensor"""
        path = self.get_temp_path("data.wav")
        expected = get_wav_data(dtype, 4, normalize=False)[::2, ::2]

        data = expected.clone()
        sox_io_backend.save(path, data, 8000)

        self.assertEqual(data, expected)


@skipIfNoSox
class TestSaveNonExistingDirectory(PytorchTestCase):
    def test_save_fail(self):
        """
        When attempted to save into a non-existing dir, error message must contain the file path.
        """
        path = os.path.join("non_existing_directory", "foo.wav")
        with self.assertRaisesRegex(RuntimeError, path):
            sox_io_backend.save(path, torch.zeros(1, 1), 8000)
