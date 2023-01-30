from unittest.mock import patch

import torch
from parameterized import parameterized
from torchaudio.io import play_audio, StreamWriter
from torchaudio_unittest.common_utils import get_sinusoid, skipIfNoAudioDevice, skipIfNoMacOS, TorchaudioTestCase


@skipIfNoAudioDevice
@skipIfNoMacOS
class PlaybackInterfaceTest(TorchaudioTestCase):
    @parameterized.expand([("uint8",), ("int16",), ("int32",), ("int64",), ("float32",), ("float64",)])
    @patch.object(StreamWriter, "write_audio_chunk")
    def test_playaudio(self, dtype, writeaudio_mock):
        """Test playaudio function.
        The patch object is used to check if the data is written
        to the output device stream, without playing the actual audio.
        """
        dtype = getattr(torch, dtype)
        sample_rate = 8000
        waveform = get_sinusoid(
            frequency=440,
            sample_rate=sample_rate,
            duration=1,  # seconds
            n_channels=1,
            dtype=dtype,
            device="cpu",
            channels_first=False,
        )

        play_audio(waveform, sample_rate=sample_rate)

        writeaudio_mock.assert_called()

    @parameterized.expand(
        [
            # Invalid number of dimensions (!= 2)
            ("int16", 1, "audiotoolbox"),
            ("int16", 3, "audiotoolbox"),
            # Invalid tensor type
            ("complex64", 2, "audiotoolbox"),
            # Invalid output device
            ("int16", 2, "audiotool"),
        ]
    )
    @patch.object(StreamWriter, "write_audio_chunk")
    def test_playaudio_invalid_options(self, dtype, ndim, device, writeaudio_mock):
        """Test playaudio function raises error with invalid options."""
        dtype = getattr(torch, dtype)
        sample_rate = 8000
        waveform = get_sinusoid(
            frequency=440,
            sample_rate=sample_rate,
            duration=1,  # seconds
            n_channels=1,
            dtype=dtype,
            device="cpu",
            channels_first=False,
        ).squeeze()

        for _ in range(ndim - 1):
            waveform = waveform.unsqueeze(-1)

        with self.assertRaises(ValueError):
            play_audio(waveform, sample_rate=sample_rate, device=device)
