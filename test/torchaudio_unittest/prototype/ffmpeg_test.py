import torch
from parameterized import parameterized
from PIL import Image
from torchaudio.prototype import ffmpeg
from torchaudio_unittest.common_utils import (
    TorchaudioTestCase,
    TempDirMixin,
    get_asset_path,
    get_wav_data,
    save_wav,
    nested_params,
)

path = get_asset_path("bipbopall.mp4")


class FFmpegLoadTest(TempDirMixin, TorchaudioTestCase):
    @nested_params(
        ["int16", "uint8", "int32"],  # "float", "double", "int64"]
        [1, 2, 4, 8],
    )
    def test_load_wav(self, dtype, num_channels):
        expected = get_wav_data(
            dtype,
            num_channels=num_channels,
            normalize=False,
            num_frames=15,
            channels_first=True,
        )
        sample_rate = 8000
        path = self.get_temp_path("test.wav")
        save_wav(path, expected, sample_rate)

        output, sr = ffmpeg.load(path)

        assert sample_rate == sr
        self.assertEqual(output.T, expected)


class FFmpegStreamerTest(TempDirMixin, TorchaudioTestCase):
    def test_info(self):
        for sinfo in ffmpeg.info(path):
            print(sinfo)

    def test_src_info(self):
        s = ffmpeg.Streamer(path)
        for i in range(s.num_src_streams):
            print(s.get_src_stream_info(i))

    def test_out_info(self):
        s = ffmpeg.Streamer(path)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, dtype=None)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, sample_rate=8000)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, dtype=torch.int16)
        s.add_basic_video_stream(1, frames_per_chunk=-1, format=None)
        s.add_basic_video_stream(1, frames_per_chunk=-1, width=3, height=5)
        s.add_basic_video_stream(1, frames_per_chunk=-1, frame_rate=7)
        s.add_basic_video_stream(1, frames_per_chunk=-1, format="BGR")

        sinfo = [s.get_out_stream_info(i) for i in range(s.num_out_streams)]
        assert sinfo[0].source_index == 0
        assert sinfo[0].filter_description == ""

        assert sinfo[1].source_index == 0
        assert "aresample=8000" in sinfo[1].filter_description

        assert sinfo[2].source_index == 0
        assert "aformat=sample_fmts=s16" in sinfo[2].filter_description

        assert sinfo[3].source_index == 1
        assert sinfo[3].filter_description == ""

        assert sinfo[4].source_index == 1
        assert "scale=width=3:height=5" in sinfo[4].filter_description

        assert sinfo[5].source_index == 1
        assert "fps=7" in sinfo[5].filter_description

        assert sinfo[6].source_index == 1
        assert "format=pix_fmts=bgr24" in sinfo[6].filter_description

    def test_modify_stream(self):
        s = ffmpeg.Streamer(path)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, sample_rate=8000)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, sample_rate=16000)
        s.add_basic_video_stream(1, frames_per_chunk=-1, width=32, height=32, frame_rate=30)
        s.add_basic_video_stream(1, frames_per_chunk=-1, width=16, height=16, frame_rate=10)
        s.remove_stream(3)
        s.remove_stream(2)
        s.remove_stream(1)
        s.remove_stream(0)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, sample_rate=8000)
        s.remove_stream(0)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, sample_rate=8000)
        # TODO: Add invalid operations

    def _save_wav(self, data, sample_rate):
        path = self.get_temp_path("test.wav")
        save_wav(path, data, sample_rate)
        return path

    def _test_wav(self, path, original, dtype):
        s = ffmpeg.Streamer(path)
        s.add_basic_audio_stream(0, frames_per_chunk=-1, dtype=dtype)
        s.process_all_packets()
        output = s.pop_chunks()[0].T
        self.assertEqual(original, output)

    @nested_params(
        ["int16", "uint8", "int32"],  # "float", "double", "int64"]
    )
    def test_wav_dtypes(self, dtype):
        original = get_wav_data(dtype, num_channels=1, normalize=False, num_frames=15, channels_first=True)
        path = self._save_wav(original, 8000)
        # provide the matching dtype
        self._test_wav(path, original, getattr(torch, dtype))
        # use the internal dtype ffmpeg picks
        self._test_wav(path, original, None)

    @nested_params([2, 4, 8])
    def test_wav_multichannels(self, num_channels):
        dtype = torch.int16
        original = torch.randint(low=-32768, high=32767, size=[num_channels, 256], dtype=dtype)
        path = self._save_wav(original, 8000)
        # provide the matching dtype
        self._test_wav(path, original, dtype)
        # use the internal dtype ffmpeg picks
        self._test_wav(path, original, None)

    def test_audio_stream(self):
        original = torch.randint(low=-32768, high=32767, size=[3, 15], dtype=torch.int16)
        path = self._save_wav(original, sample_rate=8000)
        expected = torch.flip(original, dims=(1,))

        s = ffmpeg.Streamer(path)
        s.add_audio_stream(0, frames_per_chunk=-1, filter_desc="areverse")
        s.process_all_packets()
        output = s.pop_chunks()[0].T
        self.assertEqual(expected, output)

    def test_audio_seek(self):
        original = torch.randint(low=-32768, high=32767, size=[3, 30], dtype=torch.int16)
        path = self._save_wav(original, sample_rate=1)

        for t in range(10, 20):
            expected = original[:, t:]

            s = ffmpeg.Streamer(path)
            s.add_basic_audio_stream(0, frames_per_chunk=-1, dtype=torch.int16)
            s.seek(t)
            s.process_all_packets()
            output = s.pop_chunks()[0].T
            self.assertEqual(expected, output)

    @parameterized.expand(
        [
            (18, 6, 3),  # num_frames is divisible by frames_per_chunk
            (18, 5, 4),  # num_frames is not divisible by frames_per_chunk
            (18, 32, 1),  # num_frames is shorter than frames_per_chunk
        ]
    )
    def test_audio_frames_per_chunk(self, num_frames, frames_per_chunk, num_chunks):
        original = torch.randint(low=-32768, high=32767, size=[3, num_frames], dtype=torch.int16)
        path = self._save_wav(original, sample_rate=8000)

        s = ffmpeg.Streamer(path)
        s.add_basic_audio_stream(0, frames_per_chunk=frames_per_chunk, num_chunks=num_chunks, dtype=torch.int16)
        i, outputs = 0, []
        for chunks in s:
            output = chunks[0].T
            outputs.append(output)
            self.assertEqual(original[:, frames_per_chunk * i : frames_per_chunk * (i + 1)], output)
            i += 1
        assert i == num_frames // frames_per_chunk + (1 if num_frames % frames_per_chunk else 0)
        self.assertEqual(torch.cat(outputs, 1), original)

    def _save_png(self, data):
        path = self.get_temp_path("test.png")
        img = Image.fromarray(data.numpy())
        img.save(path)
        return path

    def _test_png(self, path, original, format):
        s = ffmpeg.Streamer(path)
        s.add_basic_video_stream(0, frames_per_chunk=-1, format=format)
        s.process_all_packets()
        output = s.pop_chunks()[0]
        self.assertEqual(original, output)

    def test_png_gray(self):
        h, w = 111, 250
        original = torch.arange(h * w, dtype=torch.int64).reshape(h, w) % 256
        original = original.to(torch.uint8)
        path = self._save_png(original)
        expected = original[None, None, ...]
        self._test_png(path, expected, format=None)

    def test_png_color(self):
        # TODO:
        # Add test with alpha channel (RGBA, ARGB, BGRA, ABGR)
        c, h, w = 3, 111, 250
        original = torch.arange(c * h * w, dtype=torch.int64).reshape(c, h, w) % 256
        original = original.to(torch.uint8)
        path = self._save_png(original.permute(1, 2, 0))
        expected = original[None, ...]
        self._test_png(path, expected, format=None)

    @parameterized.expand(
        [
            ("hflip", 2),
            ("vflip", 1),
        ]
    )
    def test_video_stream(self, filter_desc, index):
        c, h, w = 3, 111, 250
        original = torch.arange(c * h * w, dtype=torch.int64).reshape(c, h, w) % 256
        original = original.to(torch.uint8)
        path = self._save_png(original.permute(1, 2, 0))
        expected = torch.flip(original, dims=(index,))[None, ...]

        s = ffmpeg.Streamer(path)
        s.add_video_stream(0, frames_per_chunk=-1, filter_desc=filter_desc)
        s.process_all_packets()
        output = s.pop_chunks()[0]
        print("expected", expected)
        print("output", output)
        self.assertEqual(expected, output)
