from .download import download_asset
import scipy.io.wavfile as wavfile
import torch


def _load(file_audio, normalize=True):
    sample_rate, waveform = wavfile.read(file_audio)
    if len(waveform.shape) == 1:
        waveform = waveform[None,:]
    else:
        waveform = waveform.T
    waveform = torch.from_numpy(waveform)
    if normalize:
        waveform = waveform.float()
    return waveform, sample_rate

__all__ = [
    "download_asset",
]
