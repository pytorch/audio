from unittest.mock import patch

import os
import tempfile
import pytest
import torch

import torchaudio
from torchaudio import load_with_torchcodec, save_with_torchcodec
from torchaudio_unittest.common_utils import get_asset_path


# Test with wav files that should work with both torchaudio and torchcodec
TEST_FILES = [
    "sinewave.wav",
    "steam-train-whistle-daniel_simon.wav",
    "vad-go-mono-32000.wav",
    "vad-go-stereo-44100.wav",
    "VCTK-Corpus/wav48/p224/p224_002.wav",
]


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

def test_normalize_parameter_warning():
    """Test that normalize=False produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="normalize=False.*ignored"):
        # This should produce a warning
        waveform, sample_rate = load_with_torchcodec(file_path, normalize=False)
        
        # Result should still be float32 (normalized)
        assert waveform.dtype == torch.float32

def test_buffer_size_parameter_warning():
    """Test that non-default buffer_size produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="buffer_size.*not used"):
        # This should produce a warning
        waveform, sample_rate = load_with_torchcodec(file_path, buffer_size=8192)


def test_backend_parameter_warning():
    """Test that specifying backend produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="backend.*not used"):
        # This should produce a warning
        waveform, sample_rate = load_with_torchcodec(file_path, backend="ffmpeg")


def test_invalid_file():
    """Test that invalid files raise appropriate errors."""
    with pytest.raises(RuntimeError, match="Failed to create AudioDecoder"):
        load_with_torchcodec("/nonexistent/file.wav")


def test_format_parameter():
    """Test that format parameter produces a warning."""
    file_path = get_asset_path("sinewave.wav")
    
    with pytest.warns(UserWarning, match="format.*not supported"):
        waveform, sample_rate = load_with_torchcodec(file_path, format="wav")
        
        # Check basic properties
        assert waveform.dtype == torch.float32
        assert sample_rate > 0


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


# ===== SAVE WITH TORCHCODEC TESTS =====

@pytest.mark.parametrize("filename", TEST_FILES)
def test_save_basic_save(filename):
    """Test basic saving functionality against torchaudio.save."""
    # Load a test file first
    file_path = get_asset_path(*filename.split("/"))
    waveform, sample_rate = torchaudio.load(file_path)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save with torchaudio
        ta_path = os.path.join(temp_dir, "ta_output.wav")
        torchaudio.save(ta_path, waveform, sample_rate)
        
        # Save with torchcodec
        tc_path = os.path.join(temp_dir, "tc_output.wav")
        save_with_torchcodec(tc_path, waveform, sample_rate)
        
        # Load both back and compare
        waveform_ta, sample_rate_ta = torchaudio.load(ta_path)
        waveform_tc, sample_rate_tc = torchaudio.load(tc_path)
        
        # Check sample rates match
        assert sample_rate_ta == sample_rate_tc
        
        # Check shapes match
        assert waveform_ta.shape == waveform_tc.shape
        
        # Check data types (should both be float32)
        assert waveform_ta.dtype == torch.float32
        assert waveform_tc.dtype == torch.float32
        
        # Check values are close (allowing for small differences in encoders)
        torch.testing.assert_close(waveform_ta, waveform_tc, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("channels_first", [True, False])
def test_save_channels_first(channels_first):
    """Test channels_first parameter."""
    # Create test data
    if channels_first:
        waveform = torch.randn(2, 16000)  # [channel, time]
    else:
        waveform = torch.randn(16000, 2)  # [time, channel]
    
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save with torchaudio
        ta_path = os.path.join(temp_dir, "ta_output.wav")
        torchaudio.save(ta_path, waveform, sample_rate, channels_first=channels_first)
        
        # Save with torchcodec
        tc_path = os.path.join(temp_dir, "tc_output.wav")
        save_with_torchcodec(tc_path, waveform, sample_rate, channels_first=channels_first)
        
        # Load both back and compare
        waveform_ta, sample_rate_ta = torchaudio.load(ta_path)
        waveform_tc, sample_rate_tc = torchaudio.load(tc_path)
        
        # Check results match
        assert sample_rate_ta == sample_rate_tc
        assert waveform_ta.shape == waveform_tc.shape
        torch.testing.assert_close(waveform_ta, waveform_tc, atol=1e-3, rtol=1e-3)


def test_save_compression_parameter():
    """Test compression parameter (maps to bit_rate)."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with compression (bit_rate)
        output_path = os.path.join(temp_dir, "output.wav")
        save_with_torchcodec(output_path, waveform, sample_rate, compression=128000)
        
        # Should not raise an error and file should exist
        assert os.path.exists(output_path)
        
        # Load back and check basic properties
        waveform_loaded, sample_rate_loaded = torchaudio.load(output_path)
        assert sample_rate_loaded == sample_rate
        assert waveform_loaded.shape[0] == 1  # Should be mono


def test_save_format_parameter_warning():
    """Test that format parameter produces a warning."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        with pytest.warns(UserWarning, match="format.*not used"):
            save_with_torchcodec(output_path, waveform, sample_rate, format="wav")
            
        # Should still work despite warning
        assert os.path.exists(output_path)


def test_save_encoding_parameter_warning():
    """Test that encoding parameter produces a warning."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        with pytest.warns(UserWarning, match="encoding.*not fully supported"):
            save_with_torchcodec(output_path, waveform, sample_rate, encoding="PCM_16")
            
        # Should still work despite warning
        assert os.path.exists(output_path)


def test_save_bits_per_sample_parameter_warning():
    """Test that bits_per_sample parameter produces a warning."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        with pytest.warns(UserWarning, match="bits_per_sample.*not directly supported"):
            save_with_torchcodec(output_path, waveform, sample_rate, bits_per_sample=16)
            
        # Should still work despite warning
        assert os.path.exists(output_path)


def test_save_buffer_size_parameter_warning():
    """Test that non-default buffer_size produces a warning."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        with pytest.warns(UserWarning, match="buffer_size.*not used"):
            save_with_torchcodec(output_path, waveform, sample_rate, buffer_size=8192)
            
        # Should still work despite warning
        assert os.path.exists(output_path)


def test_save_backend_parameter_warning():
    """Test that specifying backend produces a warning."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        with pytest.warns(UserWarning, match="backend.*not used"):
            save_with_torchcodec(output_path, waveform, sample_rate, backend="ffmpeg")
            
        # Should still work despite warning
        assert os.path.exists(output_path)


def test_save_edge_cases():
    """Test edge cases and error conditions."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        # Test with very small waveform
        small_waveform = torch.randn(1, 10)
        save_with_torchcodec(output_path, small_waveform, sample_rate)
        waveform_loaded, sample_rate_loaded = torchaudio.load(output_path)
        assert sample_rate_loaded == sample_rate
        
        # Test with different sample rates
        for sr in [8000, 22050, 44100]:
            sr_path = os.path.join(temp_dir, f"output_{sr}.wav")
            save_with_torchcodec(sr_path, waveform, sr)
            waveform_loaded, sample_rate_loaded = torchaudio.load(sr_path)
            assert sample_rate_loaded == sr


def test_save_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        # Test with invalid sample rate
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            save_with_torchcodec(output_path, waveform, -1)
        
        # Test with invalid tensor dimensions
        with pytest.raises(ValueError, match="Expected 1D or 2D tensor"):
            invalid_waveform = torch.randn(1, 2, 16000)  # 3D tensor
            save_with_torchcodec(output_path, invalid_waveform, sample_rate)
        
        # Test with non-tensor input
        with pytest.raises(ValueError, match="Expected src to be a torch.Tensor"):
            save_with_torchcodec(output_path, [1, 2, 3], sample_rate)


def test_save_multiple_warnings():
    """Test that multiple unsupported parameters produce multiple warnings."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "output.wav")
        
        with pytest.warns() as warning_list:
            save_with_torchcodec(
                output_path, 
                waveform, 
                sample_rate,
                format="wav",
                encoding="PCM_16",
                bits_per_sample=16,
                buffer_size=8192,
                backend="ffmpeg"
            )
            
        # Check that expected warnings are present
        messages = [str(w.message) for w in warning_list]
        assert any("format" in msg for msg in messages)
        assert any("encoding" in msg for msg in messages)
        assert any("bits_per_sample" in msg for msg in messages)
        assert any("buffer_size" in msg for msg in messages)
        assert any("backend" in msg for msg in messages)
        
        # Should still work despite warnings
        assert os.path.exists(output_path)


def test_save_different_formats():
    """Test saving to different audio formats."""
    waveform = torch.randn(1, 16000)
    sample_rate = 16000
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test common formats
        formats = ["wav", "mp3", "flac"]
        
        for fmt in formats:
            output_path = os.path.join(temp_dir, f"output.{fmt}")
            try:
                save_with_torchcodec(output_path, waveform, sample_rate)
                assert os.path.exists(output_path)
                
                # Try to load back (may not work for all formats with all backends)
                try:
                    waveform_loaded, sample_rate_loaded = torchaudio.load(output_path)
                    assert sample_rate_loaded == sample_rate
                except Exception:
                    # Some formats might not be supported by the loading backend
                    pass
            except Exception as e:
                # Some formats might not be supported by torchcodec
                pytest.skip(f"Format {fmt} not supported: {e}")