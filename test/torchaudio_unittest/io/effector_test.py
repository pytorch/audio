from parameterized import parameterized

from torchaudio.io import AudioEffector
from torchaudio_unittest.common_utils import get_sinusoid, skipIfNoFFmpeg, TorchaudioTestCase

from .common import lt42


@skipIfNoFFmpeg
class EffectorTest(TorchaudioTestCase):
    def test_null(self):
        """No effect and codec will return the same result"""
        sample_rate = 8000
        frames_per_chunk = 256

        effector = AudioEffector(effect=None, format=None)
        original = get_sinusoid(n_channels=3, sample_rate=sample_rate, channels_first=False)

        # one-go
        output = effector.apply(original, sample_rate)
        self.assertEqual(original, output)
        # streaming
        for i, chunk in enumerate(effector.stream(original, sample_rate, frames_per_chunk)):
            start = i * frames_per_chunk
            end = (i + 1) * frames_per_chunk
            self.assertEqual(original[start:end, :], chunk)

    @parameterized.expand(
        [
            ("ogg", "flac"),  # flac only supports s16 and s32
            ("ogg", "opus"),  # opus only supports 48k Hz
            ("ogg", "vorbis"),  # vorbis only supports stereo
            # ("ogg", "vorbis", 44100),
            # this fails with small descrepancy; 441024 vs 441000
            # TODO: investigate
            ("wav", None),
            ("wav", "pcm_u8"),
            ("mp3", None),
            ("mulaw", None, 44100),  # mulaw is encoded without header
        ]
    )
    def test_formats(self, format, encoder, sample_rate=8000):
        """Formats (some with restrictions) just work without an issue in effector"""

        effector = AudioEffector(format=format, encoder=encoder)
        original = get_sinusoid(n_channels=3, sample_rate=sample_rate, channels_first=False)

        output = effector.apply(original, sample_rate)

        # On 4.1 OPUS produces 8020 samples (extra 20)
        # this has been fixed on 4.2+
        if encoder == "opus" and lt42():
            return

        self.assertEqual(original.shape, output.shape)

        # Note
        # MP3 adds padding which cannot be removed when the encoded data is written to
        # file-like object without seek method.
        # The number of padding is retrievable as `AVCoedcContext::initial_padding`
        # https://ffmpeg.org/doxygen/4.1/structAVCodecContext.html#a8f95550ce04f236e9915516d04d3d1ab
        # but this is not exposed yet.
        # These "priming" samples have negative time stamp, so we can also add logic
        # to discard them at decoding, however, as far as I checked, when data is loaded
        # with StreamReader, the time stamp is reset. I tried options like avoid_negative_ts,
        # https://ffmpeg.org/ffmpeg-formats.html
        # but it made no difference. Perhaps this is because the information about negative
        # timestamp is only available at encoding side, and it presumably is written to
        # header file, but it is not happening somehow with file-like object.
        # Need to investigate more to remove MP3 padding
        if format == "mp3":
            return

        for chunk in effector.stream(original, sample_rate, frames_per_chunk=original.size(0)):
            self.assertEqual(original.shape, chunk.shape)

    @parameterized.expand([("loudnorm=I=-16:LRA=11:TP=-1.5",), ("volume=2",)])
    def test_effect(self, effect):
        sample_rate = 8000

        effector = AudioEffector(effect=effect)
        original = get_sinusoid(n_channels=3, sample_rate=sample_rate, channels_first=False)

        output = effector.apply(original, sample_rate)
        self.assertEqual(original.shape, output.shape)

    def test_resample(self):
        """Resample option allows to change the sampling rate"""
        sample_rate = 8000
        output_sample_rate = 16000
        num_channels = 3

        effector = AudioEffector(effect="lowpass")
        original = get_sinusoid(n_channels=num_channels, sample_rate=sample_rate, channels_first=False)

        output = effector.apply(original, sample_rate, output_sample_rate)
        self.assertEqual(output.shape, [output_sample_rate, num_channels])

        for chunk in effector.stream(
            original, sample_rate, output_sample_rate=output_sample_rate, frames_per_chunk=output_sample_rate
        ):
            self.assertEqual(chunk.shape, [output_sample_rate, num_channels])
