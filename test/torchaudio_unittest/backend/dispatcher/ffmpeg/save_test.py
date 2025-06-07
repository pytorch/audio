import io
import os
import pathlib
import subprocess
import sys
from functools import partial
from typing import Optional

import torch
from parameterized import parameterized
from torchaudio._backend.ffmpeg import _parse_save_args
from torchaudio._backend.utils import get_save_func
from torchaudio.io import CodecConfig

from torchaudio_unittest.backend.dispatcher.sox.common import get_enc_params, name_func
from torchaudio_unittest.common_utils import (
    disabledInCI,
    get_wav_data,
    load_wav,
    nested_params,
    PytorchTestCase,
    save_wav,
    skipIfNoExec,
    skipIfNoFFmpeg,
    TempDirMixin,
    TorchaudioTestCase,
)


def _convert_audio_file(src_path, dst_path, muxer=None, encoder=None, sample_fmt=None):
    command = ["ffmpeg", "-hide_banner", "-y", "-i", src_path, "-strict", "-2"]
    if muxer:
        command += ["-f", muxer]
    if encoder:
        command += ["-acodec", encoder]
    if sample_fmt:
        command += ["-sample_fmt", sample_fmt]
    command += [dst_path]
    print(" ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)


class SaveTestBase(TempDirMixin, TorchaudioTestCase):
    _save = staticmethod(partial(get_save_func(), backend="ffmpeg"))

    def assert_save_consistency(
        self,
        format: str,
        *,
        compression: Optional[CodecConfig] = None,
        encoding: str = None,
        bits_per_sample: int = None,
        sample_rate: float = 8000,
        num_channels: int = 2,
        num_frames: float = 3 * 8000,
        src_dtype: str = "int32",
        test_mode: str = "path",
    ):
        """`save` function produces file that is comparable with `ffmpeg` command

        To compare that the file produced by `save` function agains the file produced by
        the equivalent `ffmpeg` command, we need to load both files.
        But there are many formats that cannot be opened with common Python modules (like
        SciPy).
        So we use `ffmpeg` command to prepare the original data and convert the saved files
        into a format that SciPy can read (PCM wav).
        The following diagram illustrates this process. The difference is 2.1. and 3.1.

        This assumes that
         - loading data with SciPy preserves the data well.
         - converting the resulting files into WAV format with `ffmpeg` preserve the data well.

                          x
                          | 1. Generate source wav file with SciPy
                          |
                          v
          -------------- wav ----------------
         |                                   |
         | 2.1. load with scipy              | 3.1. Convert to the target
         |   then save it into the target    |      format depth with ffmpeg
         |   format with torchaudio          |
         v                                   v
        target format                       target format
         |                                   |
         | 2.2. Convert to wav with ffmpeg   | 3.2. Convert to wav with ffmpeg
         |                                   |
         v                                   v
        wav                                 wav
         |                                   |
         | 2.3. load with scipy              | 3.3. load with scipy
         |                                   |
         v                                   v
        tensor -------> compare <--------- tensor

        """
        src_path = self.get_temp_path("1.source.wav")
        tgt_path = self.get_temp_path(f"2.1.torchaudio.{format}")
        tst_path = self.get_temp_path("2.2.result.wav")
        sox_path = self.get_temp_path(f"3.1.ffmpeg.{format}")
        ref_path = self.get_temp_path("3.2.ref.wav")

        # 1. Generate original wav
        data = get_wav_data(src_dtype, num_channels, normalize=False, num_frames=num_frames)
        save_wav(src_path, data, sample_rate)

        # 2.1. Convert the original wav to target format with torchaudio
        data = load_wav(src_path, normalize=False)[0]
        if test_mode == "path":
            ext = format
            self._save(
                tgt_path,
                data,
                sample_rate,
                compression=compression,
                format=format,
                encoding=encoding,
                bits_per_sample=bits_per_sample,
            )
        elif test_mode == "fileobj":
            ext = None
            with open(tgt_path, "bw") as file_:
                self._save(
                    file_,
                    data,
                    sample_rate,
                    compression=compression,
                    format=format,
                    encoding=encoding,
                    bits_per_sample=bits_per_sample,
                )
        elif test_mode == "bytesio":
            file_ = io.BytesIO()
            ext = None
            self._save(
                file_,
                data,
                sample_rate,
                compression=compression,
                format=format,
                encoding=encoding,
                bits_per_sample=bits_per_sample,
            )
            file_.seek(0)
            with open(tgt_path, "bw") as f:
                f.write(file_.read())
        else:
            raise ValueError(f"Unexpected test mode: {test_mode}")
        # 2.2. Convert the target format to wav with ffmpeg
        _convert_audio_file(tgt_path, tst_path, encoder="pcm_f32le")
        # 2.3. Load with SciPy
        found = load_wav(tst_path, normalize=False)[0]

        # 3.1. Convert the original wav to target format with ffmpeg
        muxer, encoder, sample_fmt = _parse_save_args(ext, format, encoding, bits_per_sample)
        _convert_audio_file(src_path, sox_path, muxer=muxer, encoder=encoder, sample_fmt=sample_fmt)
        # 3.2. Convert the target format to wav with ffmpeg
        _convert_audio_file(sox_path, ref_path, encoder="pcm_f32le")
        # 3.3. Load with SciPy
        expected = load_wav(ref_path, normalize=False)[0]

        self.assertEqual(found, expected)


@disabledInCI
@skipIfNoExec("sox")
@skipIfNoExec("ffmpeg")
@skipIfNoFFmpeg
class SaveTest(SaveTestBase):
    def test_pathlike(self):
        """FFmpeg dispatcher can save audio data to pathlike object"""
        sample_rate = 16000
        dtype = "float32"
        num_channels = 2
        duration = 1

        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        self._save(pathlib.Path(path), data, sample_rate)

    @nested_params(
        ["path", "fileobj", "bytesio"],
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
    def test_save_wav(self, test_mode, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency("wav", encoding=encoding, bits_per_sample=bits_per_sample, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        [
            ("float32",),
            ("int32",),
            ("int16",),
            ("uint8",),
        ],
    )
    def test_save_wav_dtype(self, test_mode, params):
        (dtype,) = params
        self.assert_save_consistency("wav", src_dtype=dtype, test_mode=test_mode)

    @nested_params(
        ["path", "fileobj", "bytesio"],
        # NOTE: Supported sample formats: s16 s32 (24 bits)
        # [8, 16, 24],
        [16, 24],
        [
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
        # -acodec flac -sample_fmt s16
        # 24 bits needs to be mapped to s32
        codec_config = CodecConfig(
            compression_level=compression_level,
        )
        self.assert_save_consistency(
            "flac", compression=codec_config, bits_per_sample=bits_per_sample, test_mode=test_mode
        )

    # @nested_params(
    #     ["path", "fileobj", "bytesio"],
    # )
    # # NOTE: FFmpeg: Unable to find a suitable output format
    # def test_save_htk(self, test_mode):
    #     self.assert_save_consistency("htk", test_mode=test_mode, num_channels=1)

    @nested_params(
        [
            None,
            -1,
            0,
            1,
            2,
            3,
            5,
            10,
        ],
        ["path", "fileobj", "bytesio"],
    )
    def test_save_vorbis(self, quality_level, test_mode):
        # NOTE: ffmpeg doesn't recognize extension "vorbis", so we use "ogg"
        # self.assert_save_consistency("vorbis", test_mode=test_mode)
        codec_config = CodecConfig(
            qscale=quality_level,
        )
        self.assert_save_consistency("ogg", compression=codec_config, test_mode=test_mode)

    # @nested_params(
    #     ["path", "fileobj", "bytesio"],
    #     [
    #         (
    #             "PCM_S",
    #             8,
    #         ),
    #         (
    #             "PCM_S",
    #             16,
    #         ),
    #         (
    #             "PCM_S",
    #             24,
    #         ),
    #         (
    #             "PCM_S",
    #             32,
    #         ),
    #         ("ULAW", 8),
    #         ("ALAW", 8),
    #         ("ALAW", 16),
    #         ("ALAW", 24),
    #         ("ALAW", 32),
    #     ],
    # )
    # NOTE: FFmpeg doesn't support encoding sphere files.
    # def test_save_sphere(self, test_mode, enc_params):
    #     encoding, bits_per_sample = enc_params
    #     self.assert_save_consistency("sph", encoding=encoding, bits_per_sample=bits_per_sample, test_mode=test_mode)

    # @nested_params(
    #     ["path", "fileobj", "bytesio"],
    #     [
    #         (
    #             "PCM_U",
    #             8,
    #         ),
    #         (
    #             "PCM_S",
    #             16,
    #         ),
    #         (
    #             "PCM_S",
    #             24,
    #         ),
    #         (
    #             "PCM_S",
    #             32,
    #         ),
    #         (
    #             "PCM_F",
    #             32,
    #         ),
    #         (
    #             "PCM_F",
    #             64,
    #         ),
    #         (
    #             "ULAW",
    #             8,
    #         ),
    #         (
    #             "ALAW",
    #             8,
    #         ),
    #     ],
    # )
    # NOTE: FFmpeg doesn't support amb.
    # def test_save_amb(self, test_mode, enc_params):
    #     encoding, bits_per_sample = enc_params
    #     self.assert_save_consistency("amb", encoding=encoding, bits_per_sample=bits_per_sample, test_mode=test_mode)

    # @nested_params(
    #     ["path", "fileobj", "bytesio"],
    # )
    # # NOTE: FFmpeg: Unable to find a suitable output format
    # def test_save_amr_nb(self, test_mode):
    #     self.assert_save_consistency("amr-nb", num_channels=1, test_mode=test_mode)

    # @nested_params(
    #     ["path", "fileobj", "bytesio"],
    # )
    # # NOTE: FFmpeg: RuntimeError: Unexpected codec: gsm
    # def test_save_gsm(self, test_mode):
    #     self.assert_save_consistency("gsm", num_channels=1, test_mode=test_mode)
    #     with self.assertRaises(RuntimeError, msg="gsm format only supports single channel audio."):
    #         self.assert_save_consistency("gsm", num_channels=2, test_mode=test_mode)
    #     with self.assertRaises(RuntimeError, msg="gsm format only supports a sampling rate of 8kHz."):
    #         self.assert_save_consistency("gsm", sample_rate=16000, test_mode=test_mode)

    @parameterized.expand(
        [
            ("wav", "PCM_S", 16),
            ("flac",),
            ("ogg",),
            # ("sph", "PCM_S", 16),
            # ("amr-nb",),
            # ("amb", "PCM_S", 16),
        ],
        name_func=name_func,
    )
    def test_save_large(self, format, encoding=None, bits_per_sample=None):
        """`self._save` can save large files."""
        sample_rate = 8000
        one_hour = 60 * 60 * sample_rate
        self.assert_save_consistency(
            format,
            # NOTE: for ogg, ffmpeg only supports >= 2 channels
            num_channels=2,
            sample_rate=8000,
            num_frames=one_hour,
            encoding=encoding,
            bits_per_sample=bits_per_sample,
        )

    @parameterized.expand(
        [
            (16,),
            # NOTE: FFmpeg doesn't support more than 16 channels.
            # (32,),
            # (64,),
            # (128,),
            # (256,),
        ],
        name_func=name_func,
    )
    def test_save_multi_channels(self, num_channels):
        """`self._save` can save audio with many channels"""
        self.assert_save_consistency("wav", encoding="PCM_S", bits_per_sample=16, num_channels=num_channels)


@skipIfNoExec("sox")
@skipIfNoFFmpeg
class TestSaveParams(TempDirMixin, PytorchTestCase):
    """Test the correctness of optional parameters of `self._save`"""

    _save = staticmethod(partial(get_save_func(), backend="ffmpeg"))

    @parameterized.expand([(True,), (False,)], name_func=name_func)
    def test_save_channels_first(self, channels_first):
        """channels_first swaps axes"""
        path = self.get_temp_path("data.wav")
        data = get_wav_data("int16", 2, channels_first=channels_first, normalize=False)
        self._save(path, data, 8000, channels_first=channels_first)
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
        self._save(path, expected, 8000, encoding=enc, bits_per_sample=bps)
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
        self._save(path, data, 8000)

        self.assertEqual(data, expected)


@disabledInCI
@skipIfNoExec("sox")
@skipIfNoFFmpeg
class TestSaveNonExistingDirectory(PytorchTestCase):
    _save = staticmethod(partial(get_save_func(), backend="ffmpeg"))

    def test_save_fail(self):
        """
        When attempted to save into a non-existing dir, error message must contain the file path.
        """
        path = os.path.join("non_existing_directory", "foo.wav")
        with self.assertRaisesRegex(RuntimeError, path):
            self._save(path, torch.zeros(1, 1), 8000)
