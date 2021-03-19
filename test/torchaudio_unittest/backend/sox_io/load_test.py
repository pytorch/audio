import io
import itertools
import tarfile

from parameterized import parameterized
from torchaudio.backend import sox_io_backend
from torchaudio._internal import module_utils as _mod_utils

from torchaudio_unittest.common_utils import (
    TempDirMixin,
    HttpServerMixin,
    PytorchTestCase,
    skipIfNoExec,
    skipIfNoModule,
    skipIfNoSox,
    get_asset_path,
    get_wav_data,
    load_wav,
    save_wav,
    sox_utils,
)
from .common import (
    name_func,
)


if _mod_utils.is_module_available("requests"):
    import requests


class LoadTestBase(TempDirMixin, PytorchTestCase):
    def assert_format(
            self,
            format: str,
            sample_rate: float,
            num_channels: int,
            compression: float = None,
            bit_depth: int = None,
            duration: float = 1,
            normalize: bool = True,
            encoding: str = None,
            atol: float = 4e-05,
            rtol: float = 1.3e-06,
    ):
        """`sox_io_backend.load` can load given format correctly.

        file encodings introduce delay and boundary effects so
        we create a reference wav file from the original file format

         x
         |
         |    1. Generate given format with Sox
         |
         v    2. Convert to wav with Sox
        given format ----------------------> wav
         |                                   |
         |    3. Load with torchaudio        | 4. Load with scipy
         |                                   |
         v                                   v
        tensor ----------> x <----------- tensor
                       5. Compare

        Underlying assumptions are;
        i. Conversion of given format to wav with Sox preserves data.
        ii. Loading wav file with scipy is correct.

        By combining i & ii, step 2. and 4. allows to load reference given format
        data without using torchaudio
        """

        path = self.get_temp_path(f'1.original.{format}')
        ref_path = self.get_temp_path('2.reference.wav')

        # 1. Generate the given format with sox
        sox_utils.gen_audio_file(
            path, sample_rate, num_channels, encoding=encoding,
            compression=compression, bit_depth=bit_depth, duration=duration,
        )
        # 2. Convert to wav with sox
        wav_bit_depth = 32 if bit_depth == 24 else None  # for 24-bit wav
        sox_utils.convert_audio_file(path, ref_path, bit_depth=wav_bit_depth)
        # 3. Load the given format with torchaudio
        data, sr = sox_io_backend.load(path, normalize=normalize)
        # 4. Load wav with scipy
        data_ref = load_wav(ref_path, normalize=normalize)[0]
        # 5. Compare
        assert sr == sample_rate
        self.assertEqual(data, data_ref, atol=atol, rtol=rtol)

    def assert_wav(self, dtype, sample_rate, num_channels, normalize, duration):
        """`sox_io_backend.load` can load wav format correctly.

        Wav data loaded with sox_io backend should match those with scipy
        """
        path = self.get_temp_path('reference.wav')
        data = get_wav_data(dtype, num_channels, normalize=normalize, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        expected = load_wav(path, normalize=normalize)[0]
        data, sr = sox_io_backend.load(path, normalize=normalize)
        assert sr == sample_rate
        self.assertEqual(data, expected)


@skipIfNoExec('sox')
@skipIfNoSox
class TestLoad(LoadTestBase):
    """Test the correctness of `sox_io_backend.load` for various formats"""
    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [8000, 16000],
        [1, 2],
        [False, True],
    )), name_func=name_func)
    def test_wav(self, dtype, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load wav format correctly."""
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=1)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [False, True],
    )), name_func=name_func)
    def test_24bit_wav(self, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load 24bit wav format correctly. Corectly casts it to ``int32`` tensor dtype."""
        self.assert_format("wav", sample_rate, num_channels, bit_depth=24, normalize=normalize, duration=1)

    @parameterized.expand(list(itertools.product(
        ['int16'],
        [16000],
        [2],
        [False],
    )), name_func=name_func)
    def test_wav_large(self, dtype, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load large wav file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_wav(dtype, sample_rate, num_channels, normalize, two_hours)

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16', 'uint8'],
        [4, 8, 16, 32],
    )), name_func=name_func)
    def test_multiple_channels(self, dtype, num_channels):
        """`sox_io_backend.load` can load wav file with more than 2 channels."""
        sample_rate = 8000
        normalize = False
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=1)

    @parameterized.expand(list(itertools.product(
        [8000, 16000, 44100],
        [1, 2],
        [96, 128, 160, 192, 224, 256, 320],
    )), name_func=name_func)
    def test_mp3(self, sample_rate, num_channels, bit_rate):
        """`sox_io_backend.load` can load mp3 format correctly."""
        self.assert_format("mp3", sample_rate, num_channels, compression=bit_rate, duration=1, atol=5e-05)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [128],
    )), name_func=name_func)
    def test_mp3_large(self, sample_rate, num_channels, bit_rate):
        """`sox_io_backend.load` can load large mp3 file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_format("mp3", sample_rate, num_channels, compression=bit_rate, duration=two_hours, atol=5e-05)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        list(range(9)),
    )), name_func=name_func)
    def test_flac(self, sample_rate, num_channels, compression_level):
        """`sox_io_backend.load` can load flac format correctly."""
        self.assert_format("flac", sample_rate, num_channels, compression=compression_level, bit_depth=16, duration=1)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [0],
    )), name_func=name_func)
    def test_flac_large(self, sample_rate, num_channels, compression_level):
        """`sox_io_backend.load` can load large flac file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_format(
            "flac", sample_rate, num_channels, compression=compression_level, bit_depth=16, duration=two_hours)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
        [-1, 0, 1, 2, 3, 3.6, 5, 10],
    )), name_func=name_func)
    def test_vorbis(self, sample_rate, num_channels, quality_level):
        """`sox_io_backend.load` can load vorbis format correctly."""
        self.assert_format("vorbis", sample_rate, num_channels, compression=quality_level, bit_depth=16, duration=1)

    @parameterized.expand(list(itertools.product(
        [16000],
        [2],
        [10],
    )), name_func=name_func)
    def test_vorbis_large(self, sample_rate, num_channels, quality_level):
        """`sox_io_backend.load` can load large vorbis file correctly."""
        two_hours = 2 * 60 * 60
        self.assert_format(
            "vorbis", sample_rate, num_channels, compression=quality_level, bit_depth=16, duration=two_hours)

    @parameterized.expand(list(itertools.product(
        ['96k'],
        [1, 2],
        [0, 5, 10],
    )), name_func=name_func)
    def test_opus(self, bitrate, num_channels, compression_level):
        """`sox_io_backend.load` can load opus file correctly."""
        ops_path = get_asset_path('io', f'{bitrate}_{compression_level}_{num_channels}ch.opus')
        wav_path = self.get_temp_path(f'{bitrate}_{compression_level}_{num_channels}ch.opus.wav')
        sox_utils.convert_audio_file(ops_path, wav_path)

        expected, sample_rate = load_wav(wav_path)
        found, sr = sox_io_backend.load(ops_path)

        assert sample_rate == sr
        self.assertEqual(expected, found)

    @parameterized.expand(list(itertools.product(
        [8000, 16000],
        [1, 2],
    )), name_func=name_func)
    def test_sphere(self, sample_rate, num_channels):
        """`sox_io_backend.load` can load sph format correctly."""
        self.assert_format("sph", sample_rate, num_channels, bit_depth=32, duration=1)

    @parameterized.expand(list(itertools.product(
        ['float32', 'int32', 'int16'],
        [8000, 16000],
        [1, 2],
        [False, True],
    )), name_func=name_func)
    def test_amb(self, dtype, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load amb format correctly."""
        bit_depth = sox_utils.get_bit_depth(dtype)
        encoding = sox_utils.get_encoding(dtype)
        self.assert_format(
            "amb", sample_rate, num_channels, bit_depth=bit_depth, duration=1, encoding=encoding, normalize=normalize)

    def test_amr_nb(self):
        """`sox_io_backend.load` can load amr_nb format correctly."""
        self.assert_format("amr-nb", sample_rate=8000, num_channels=1, bit_depth=32, duration=1)


@skipIfNoExec('sox')
@skipIfNoSox
class TestLoadParams(TempDirMixin, PytorchTestCase):
    """Test the correctness of frame parameters of `sox_io_backend.load`"""
    original = None
    path = None

    def setUp(self):
        super().setUp()
        sample_rate = 8000
        self.original = get_wav_data('float32', num_channels=2)
        self.path = self.get_temp_path('test.wav')
        save_wav(self.path, self.original, sample_rate)

    @parameterized.expand(list(itertools.product(
        [0, 1, 10, 100, 1000],
        [-1, 1, 10, 100, 1000],
    )), name_func=name_func)
    def test_frame(self, frame_offset, num_frames):
        """num_frames and frame_offset correctly specify the region of data"""
        found, _ = sox_io_backend.load(self.path, frame_offset, num_frames)
        frame_end = None if num_frames == -1 else frame_offset + num_frames
        self.assertEqual(found, self.original[:, frame_offset:frame_end])

    @parameterized.expand([(True, ), (False, )], name_func=name_func)
    def test_channels_first(self, channels_first):
        """channels_first swaps axes"""
        found, _ = sox_io_backend.load(self.path, channels_first=channels_first)
        expected = self.original if channels_first else self.original.transpose(1, 0)
        self.assertEqual(found, expected)


@skipIfNoSox
class TestLoadWithoutExtension(PytorchTestCase):
    def test_mp3(self):
        """Providing format allows to read mp3 without extension

        libsox does not check header for mp3

        https://github.com/pytorch/audio/issues/1040

        The file was generated with the following command
            ffmpeg -f lavfi -i "sine=frequency=1000:duration=5" -ar 16000 -f mp3 test_noext
        """
        path = get_asset_path("mp3_without_ext")
        _, sr = sox_io_backend.load(path, format="mp3")
        assert sr == 16000


class CloggedFileObj:
    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.buffer = b''

    def read(self, n):
        if not self.buffer:
            self.buffer += self.fileobj.read(n)
        ret = self.buffer[:2]
        self.buffer = self.buffer[2:]
        return ret


@skipIfNoSox
@skipIfNoExec('sox')
class TestFileObject(TempDirMixin, PytorchTestCase):
    """
    In this test suite, the result of file-like object input is compared against file path input,
    because `load` function is rigrously tested for file path inputs to match libsox's result,
    """
    @parameterized.expand([
        ('wav', None),
        ('mp3', 128),
        ('mp3', 320),
        ('flac', 0),
        ('flac', 5),
        ('flac', 8),
        ('vorbis', -1),
        ('vorbis', 10),
        ('amb', None),
    ])
    def test_fileobj(self, ext, compression):
        """Loading audio via file object returns the same result as via file path."""
        sample_rate = 16000
        format_ = ext if ext in ['mp3'] else None
        path = self.get_temp_path(f'test.{ext}')

        sox_utils.gen_audio_file(
            path, sample_rate, num_channels=2,
            compression=compression)
        expected, _ = sox_io_backend.load(path)

        with open(path, 'rb') as fileobj:
            found, sr = sox_io_backend.load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand([
        ('wav', None),
        ('mp3', 128),
        ('mp3', 320),
        ('flac', 0),
        ('flac', 5),
        ('flac', 8),
        ('vorbis', -1),
        ('vorbis', 10),
        ('amb', None),
    ])
    def test_bytesio(self, ext, compression):
        """Loading audio via BytesIO object returns the same result as via file path."""
        sample_rate = 16000
        format_ = ext if ext in ['mp3'] else None
        path = self.get_temp_path(f'test.{ext}')

        sox_utils.gen_audio_file(
            path, sample_rate, num_channels=2,
            compression=compression)
        expected, _ = sox_io_backend.load(path)

        with open(path, 'rb') as file_:
            fileobj = io.BytesIO(file_.read())
        found, sr = sox_io_backend.load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand([
        ('wav', None),
        ('mp3', 128),
        ('mp3', 320),
        ('flac', 0),
        ('flac', 5),
        ('flac', 8),
        ('vorbis', -1),
        ('vorbis', 10),
        ('amb', None),
    ])
    def test_bytesio_clogged(self, ext, compression):
        """Loading audio via clogged file object returns the same result as via file path.

        This test case validates the case where fileobject returns shorter bytes than requeted.
        """
        sample_rate = 16000
        format_ = ext if ext in ['mp3'] else None
        path = self.get_temp_path(f'test.{ext}')

        sox_utils.gen_audio_file(
            path, sample_rate, num_channels=2,
            compression=compression)
        expected, _ = sox_io_backend.load(path)

        with open(path, 'rb') as file_:
            fileobj = CloggedFileObj(io.BytesIO(file_.read()))
        found, sr = sox_io_backend.load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand([
        ('wav', None),
        ('mp3', 128),
        ('mp3', 320),
        ('flac', 0),
        ('flac', 5),
        ('flac', 8),
        ('vorbis', -1),
        ('vorbis', 10),
        ('amb', None),
    ])
    def test_bytesio_tiny(self, ext, compression):
        """Loading very small audio via file object returns the same result as via file path.
        """
        sample_rate = 16000
        format_ = ext if ext in ['mp3'] else None
        path = self.get_temp_path(f'test.{ext}')

        sox_utils.gen_audio_file(
            path, sample_rate, num_channels=2,
            compression=compression, duration=1 / 1600)
        expected, _ = sox_io_backend.load(path)

        with open(path, 'rb') as file_:
            fileobj = io.BytesIO(file_.read())
        found, sr = sox_io_backend.load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand([
        ('wav', None),
        ('mp3', 128),
        ('mp3', 320),
        ('flac', 0),
        ('flac', 5),
        ('flac', 8),
        ('vorbis', -1),
        ('vorbis', 10),
        ('amb', None),
    ])
    def test_tarfile(self, ext, compression):
        """Loading compressed audio via file-like object returns the same result as via file path."""
        sample_rate = 16000
        format_ = ext if ext in ['mp3'] else None
        audio_file = f'test.{ext}'
        audio_path = self.get_temp_path(audio_file)
        archive_path = self.get_temp_path('archive.tar.gz')

        sox_utils.gen_audio_file(
            audio_path, sample_rate, num_channels=2,
            compression=compression)
        expected, _ = sox_io_backend.load(audio_path)

        with tarfile.TarFile(archive_path, 'w') as tarobj:
            tarobj.add(audio_path, arcname=audio_file)
        with tarfile.TarFile(archive_path, 'r') as tarobj:
            fileobj = tarobj.extractfile(audio_file)
            found, sr = sox_io_backend.load(fileobj, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)


@skipIfNoSox
@skipIfNoExec('sox')
@skipIfNoModule("requests")
class TestFileObjectHttp(HttpServerMixin, PytorchTestCase):
    @parameterized.expand([
        ('wav', None),
        ('mp3', 128),
        ('mp3', 320),
        ('flac', 0),
        ('flac', 5),
        ('flac', 8),
        ('vorbis', -1),
        ('vorbis', 10),
        ('amb', None),
    ])
    def test_requests(self, ext, compression):
        sample_rate = 16000
        format_ = ext if ext in ['mp3'] else None
        audio_file = f'test.{ext}'
        audio_path = self.get_temp_path(audio_file)

        sox_utils.gen_audio_file(
            audio_path, sample_rate, num_channels=2, compression=compression)
        expected, _ = sox_io_backend.load(audio_path)

        url = self.get_url(audio_file)
        with requests.get(url, stream=True) as resp:
            found, sr = sox_io_backend.load(resp.raw, format=format_)

        assert sr == sample_rate
        self.assertEqual(expected, found)

    @parameterized.expand(list(itertools.product(
        [0, 1, 10, 100, 1000],
        [-1, 1, 10, 100, 1000],
    )), name_func=name_func)
    def test_frame(self, frame_offset, num_frames):
        """num_frames and frame_offset correctly specify the region of data"""
        sample_rate = 8000
        audio_file = 'test.wav'
        audio_path = self.get_temp_path(audio_file)

        original = get_wav_data('float32', num_channels=2)
        save_wav(audio_path, original, sample_rate)
        frame_end = None if num_frames == -1 else frame_offset + num_frames
        expected = original[:, frame_offset:frame_end]

        url = self.get_url(audio_file)
        with requests.get(url, stream=True) as resp:
            found, sr = sox_io_backend.load(resp.raw, frame_offset, num_frames)

        assert sr == sample_rate
        self.assertEqual(expected, found)
