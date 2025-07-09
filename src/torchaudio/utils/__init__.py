from .download import download_asset
import scipy.io.wavfile as wavfile
import torch

from torchcodec.decoders import AudioDecoder

def load_torchcodec(file, **args):
    decoder = AudioDecoder(file)
    if 'start_seconds' in args or 'stop_seconds' in args:
        samples = decoder.get_samples_played_in_range(**args)
    else:
        samples = decoder.get_all_samples()
    return (samples.data, samples.sample_rate)

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
    "load_torchcodec",
    "download_asset",
]
