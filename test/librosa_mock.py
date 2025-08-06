import re
import os
from pathlib import Path
import torch

def mock_function(f):
    """
    Create a mocked version of a function from the librosa library that loads a precomputed result
    if it exists, otherwise computes the result and saves it for future use.
    """
    prefix = "torchaudio_unittest/assets/librosa_expected_results/"
    def wrapper(request, *args, **kwargs):
        if request is not None:
            if os.path.exists(f"{prefix}{request}.pt"):
                return torch.load(f"{prefix}{request}.pt", weights_only=False)
        import librosa
        result = eval(f)(*args, **kwargs)
        if request is not None:
            path = Path(f"{prefix}{request}.pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, path)
        return result
    return wrapper

griffinlim = mock_function("librosa.griffinlim")

mel = mock_function("librosa.filters.mel")

power_to_db = mock_function("librosa.core.power_to_db")

amplitude_to_db = mock_function("librosa.core.amplitude_to_db")

phase_vocoder = mock_function("librosa.phase_vocoder")

spectrogram = mock_function("librosa.core.spectrum._spectrogram")

mel_spectrogram = mock_function("librosa.feature.melspectrogram")

mfcc = mock_function("librosa.feature.mfcc")

spectral_centroid = mock_function("librosa.feature.spectral_centroid")
