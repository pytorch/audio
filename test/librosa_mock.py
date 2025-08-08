import re
import os
from pathlib import Path
import torch

def mock_function(f):
    """
    Create a mocked version of a function from the librosa library that loads a precomputed result
    if it exists. The commented out part otherwise computes the result and saves it for future use.
    This is used to compare torchaudio functionality to the equivalent functionalty in librosa without
    depending on librosa after results are precomputed.
    """
    this_file = Path(__file__).parent.resolve()
    expected_results_folder = this_file / "torchaudio_unittest" / "assets" / "librosa_expected_results"
    def wrapper(request, *args, **kwargs):
        mocked_results = expected_results_folder / f"{request}.pt"
        # return torch.load(mocked_results, weights_only=False)

        # Old definition used for generation:
        if os.path.exists(mocked_results):
            return torch.load(mocked_results, weights_only=False)
        import librosa
        result = eval(f)(*args, **kwargs)
        if request is not None:
            mocked_results.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, mocked_results)
        return result
    return wrapper

griffinlim = mock_function("librosa.griffinlim")

mel = mock_function("librosa.filters.mel")

power_to_db = mock_function("librosa.core.power_to_db")

amplitude_to_db = mock_function("librosa.core.amplitude_to_db")

phase_vocoder = mock_function("librosa.phase_vocoder")

spectrogram = mock_function("librosa.core.spectrum._spectrogram")

mel_spectrogram = mock_function("librosa.feature.melspectrogram")

def _mfcc_from_waveform(waveform, sample_rate, n_fft, hop_length, n_mels, n_mfcc):
    import librosa
    melspec = librosa.feature.melspectrogram(
        y=waveform[0].cpu().numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        htk=True,
        norm=None,
        pad_mode="reflect",
    )
    return librosa.feature.mfcc(S=librosa.core.power_to_db(melspec), n_mfcc=n_mfcc, dct_type=2, norm="ortho")

mfcc_from_waveform = mock_function("_mfcc_from_waveform")


spectral_centroid = mock_function("librosa.feature.spectral_centroid")
