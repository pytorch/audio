from unittest.mock import patch

import pytest
import torch

import torchaudio
from torchaudio import load_with_torchcodec
from torchaudio_unittest.common_utils import get_asset_path, skipIfNoModule


# Test with wav files that should work with both torchaudio and torchcodec
TEST_FILES = [
    "sinewave.wav",
    "steam-train-whistle-daniel_simon.wav",
    "vad-go-mono-32000.wav",
    "vad-go-stereo-44100.wav",
    "VCTK-Corpus/wav48/p224/p224_002.wav",
]


@skipIfNoModule("torchcodec")
@pytest.mark.parametrize("filename", TEST_FILES)
def test_basic_load(filename):
    """Test basic loading functionality against torchaudio.load."""
    file_path = get_asset_path(*filename.split("/"))
    
    # Load with torchaudio
    waveform_ta, sample_rate_ta = torchaudio.load(file_path)
    
    # Load with torchcodec
    waveform_tc, sample_rate_tc = load_with_torchcodec(file_path)
    
    # Check sample rates match
    assert sample_rate_ta == sample_rate_tc
    
    # Check shapes match
    assert waveform_ta.shape == waveform_tc.shape
    
    # Check data types (should both be float32)
    assert waveform_ta.dtype == torch.float32
    assert waveform_tc.dtype == torch.float32
    
    # Check values are close (allowing for small differences in decoders)
    torch.testing.assert_close(waveform_ta, waveform_tc)

@skipIfNoModule("torchcodec")
@pytest.mark.parametrize("frame_offset,num_frames", [
    (0, 1000),      # First 1000 samples
    (1000, 2000),   # 2000 samples starting from 1000
    (5000, -1),     # From 5000 to end
    (0, -1),        # Full file
])
def test_frame_offset_and_num_frames(frame_offset, num_frames):
    """Test frame_offset and num_frames parameters."""
    file_path = get_asset_path("steam-train-whistle-daniel_simon.wav")
    
    # Load with torchaudio
    waveform_ta, sample_rate_ta = torchaudio.load(
        file_path, frame_offset=frame_offset, num_frames=num_frames
    )
    
    # Load with torchcodec
    waveform_tc, sample_rate_tc = load_with_torchcodec(
        file_path, frame_offset=frame_offset, num_frames=num_frames
    )
    
    # Check results match
    assert sample_rate_ta == sample_rate_tc
    assert waveform_ta.shape == waveform_tc.shape
    torch.testing.assert_close(waveform_ta, waveform_tc)

@skipIfNoModule("torchcodec")
def test_channels_first():
    """Test channels_first parameter."""
    file_path = get_asset_path("vad-go-stereo-44100.wav")  # Stereo file
    
    # Test channels_first=True (default)
    waveform_cf_true, sample_rate = load_with_torchcodec(file_path, channels_first=True)
    
    # Test channels_first=False
    waveform_cf_false, _ = load_with_torchcodec(file_path, channels_first=False)
    
    # Check that transpose relationship holds
    assert waveform_cf_true.shape == waveform_cf_false.transpose(0, 1).shape
    torch.testing.assert_close(waveform_cf_true, waveform_cf_false.transpose(0, 1))
    
    # Compare with torchaudio
    waveform_ta_true, _ = torchaudio.load(file_path, channels_first=True)
    waveform_ta_false, _ = torchaudio.load(file_path, channels_first=False)
    
    assert waveform_cf_true.shape == waveform_ta_true.shape
    assert waveform_cf_false.shape == waveform_ta_false.shape
    torch.testing.assert_close(waveform_cf_true, waveform_ta_true)
    torch.testing.assert_close(waveform_cf_false, waveform_ta_false)

@skipIfNoModule("torchcodec")
def test_normalize_parameter_warning():
    """Test that normalize=False produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="normalize=False.*ignored"):
        # This should produce a warning
        waveform, sample_rate = load_with_torchcodec(file_path, normalize=False)
        
        # Result should still be float32 (normalized)
        assert waveform.dtype == torch.float32

@skipIfNoModule("torchcodec")
def test_buffer_size_parameter_warning():
    """Test that non-default buffer_size produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="buffer_size.*not used"):
        # This should produce a warning
        waveform, sample_rate = load_with_torchcodec(file_path, buffer_size=8192)


@skipIfNoModule("torchcodec")
def test_backend_parameter_warning():
    """Test that specifying backend produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="backend.*not used"):
        # This should produce a warning
        waveform, sample_rate = load_with_torchcodec(file_path, backend="ffmpeg")


@skipIfNoModule("torchcodec")
def test_invalid_file():
    """Test that invalid files raise appropriate errors."""
    with pytest.raises(RuntimeError, match="Failed to create AudioDecoder"):
        load_with_torchcodec("/nonexistent/file.wav")


@skipIfNoModule("torchcodec")
def test_format_parameter():
    """Test that format parameter produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="format.*not supported"):
        waveform, sample_rate = load_with_torchcodec(file_path, format="wav")
        
        # Check basic properties
        assert waveform.dtype == torch.float32
        assert sample_rate > 0


@skipIfNoModule("torchcodec")
def test_multiple_warnings():
    """Test that multiple unsupported parameters produce multiple warnings."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns() as warning_list:
        # This should produce multiple warnings
        waveform, sample_rate = load_with_torchcodec(
            file_path, 
            normalize=False, 
            buffer_size=8192, 
            backend="ffmpeg"
        )
        
        
        # Check that expected warnings are present
        messages = [str(w.message) for w in warning_list]
        assert any("normalize=False" in msg for msg in messages)
        assert any("buffer_size" in msg for msg in messages)
        assert any("backend" in msg for msg in messages)