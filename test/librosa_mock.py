import librosa
import os
import torch

if os.path.exists("librosa_cache.pt"):
    CACHE = torch.load("librosa_cache.pt", weights_only=False)
else:
    import librosa
    CACHE = {}

def save_cache():
    torch.save(CACHE, "librosa_cache.pt")

def mock_function(f):
    def wrapper(request, *args, **kwargs):
        if request is not None and request in CACHE:
            return CACHE[request]
        result = f(*args, **kwargs)
        if request is not None:
            CACHE[request] = result
        return result
    return wrapper

griffinlim = mock_function(librosa.griffinlim)

mel = mock_function(librosa.filters.mel)

power_to_db = mock_function(librosa.core.power_to_db)

amplitude_to_db = mock_function(librosa.core.amplitude_to_db)

phase_vocoder = mock_function(librosa.phase_vocoder)

spectrogram = mock_function(librosa.core.spectrum._spectrogram)

mel_spectrogram = mock_function(librosa.feature.melspectrogram)

mfcc = mock_function(librosa.feature.mfcc)

spectral_centroid = mock_function(librosa.feature.spectral_centroid)
