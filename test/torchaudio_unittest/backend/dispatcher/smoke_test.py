import io

from torchaudio._backend.utils import get_info_func, get_load_func, get_save_func
from torchaudio_unittest.common_utils import get_wav_data, PytorchTestCase, skipIfNoFFmpeg, TempDirMixin


@skipIfNoFFmpeg
class SmokeTest(TempDirMixin, PytorchTestCase):
    def run_smoke_test(self, ext, sample_rate, num_channels, *, dtype="float32"):
        duration = 1
        num_frames = sample_rate * duration
        path = self.get_temp_path(f"test.{ext}")
        original = get_wav_data(dtype, num_channels, normalize=False, num_frames=num_frames)

        get_save_func()(path, original, sample_rate)
        info = get_info_func()(path)
        assert info.sample_rate == sample_rate
        assert info.num_channels == num_channels

        loaded, sr = get_load_func()(path, normalize=False)
        assert sr == sample_rate
        assert loaded.shape[0] == num_channels

    def test_wav(self):
        dtype = "float32"
        sample_rate = 16000
        num_channels = 2
        self.run_smoke_test("wav", sample_rate, num_channels, dtype=dtype)


@skipIfNoFFmpeg
class SmokeTestFileObj(TempDirMixin, PytorchTestCase):
    def run_smoke_test(self, ext, sample_rate, num_channels, *, dtype="float32"):
        buffer_size = 8192
        duration = 1
        num_frames = sample_rate * duration
        fileobj = io.BytesIO()
        original = get_wav_data(dtype, num_channels, normalize=False, num_frames=num_frames)

        get_save_func()(fileobj, original, sample_rate, format=ext, buffer_size=buffer_size)

        fileobj.seek(0)
        info = get_info_func()(fileobj, format=ext, buffer_size=buffer_size)
        assert info.sample_rate == sample_rate
        assert info.num_channels == num_channels

        fileobj.seek(0)
        loaded, sr = get_load_func()(fileobj, normalize=False, format=ext, buffer_size=buffer_size)
        assert sr == sample_rate
        assert loaded.shape[0] == num_channels

    def test_wav(self):
        dtype = "float32"
        sample_rate = 16000
        num_channels = 2
        self.run_smoke_test("wav", sample_rate, num_channels, dtype=dtype)
