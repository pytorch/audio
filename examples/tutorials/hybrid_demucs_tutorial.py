"""
Hybrid Demucs Music Separation
==============================

**Author**: `Sean Kim <https://github.com/skim0514>`__

This tutorial shows how to use the Hybrid Demucs model in order to
perform music separation

"""


######################################################################
# 1. Overview
# -----------
# 
# Performing music separation is composed of the following steps
# 
# 1. Build the Hybrid Demucs pipeline.
# 2. Format the waveform into chunks of expected sizes and loop through
#    chunks (with overlap) and feed into pipeline.
# 3. Collect output chunks and combine according to the way they have been
#    overlapped.
# 
# The Hybrid Demucs model is a developed version of the Demucs model, a
# waveform based model which succesfully separated music into its
# respective parts. Hybrid Demucs effectively uses spectrogram to learn
# through the frequency domain and also moves to time convolutions.
# 


######################################################################
# 2. Preparation
# --------------
# 
# First, we install the necessary dependencies. In addition to
# ``torchaudio``, ``mir_eval`` is required to perform Si-SDR calculations
# 

import torch
import torchaudio
import typing
from IPython.display import Audio
from torchaudio.utils import download_asset
import matplotlib.pyplot as plt
import mir_eval

try:
    import google.colab

    print(
        """
        To enable running this notebook in Google Colab, install nightly
        torch and torchaudio builds and the requisite third party libraries by
        adding the following code block to the top of the notebook before running it:

        !pip3 uninstall -y torch torchvision torchaudio
        !pip3 install --pre torch torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
        !pip3 install mir_eval
        """
    )
except ModuleNotFoundError:
    pass

print(torch.__version__)
print(torchaudio.__version__)



######################################################################
# 3. Construct the pipeline
# -------------------------
# 
# Pre-trained model weights and related pipeline components are bundled as
# :py:func:`torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`. This is a
# HDemucs model trained on
# `MUSDB18-HQ <https://zenodo.org/record/3338373>`__ and other
# alternative sources.
# 

from torchaudio.prototype import pipelines
bundle = pipelines.HDEMUCS_HIGH_MUSDB_PLUS

model = bundle.get_model()

sample_rate = bundle.sample_rate

print(f"Sample rate: {sample_rate}")


######################################################################
# 4. Configure the application function
# -------------------------------------
# 
# Due to the nature of the model, it is very difficult to have sufficient
# memory to split an entire song at once. As a result, to split a full
# song, the song must be chunked into smaller segments and ran through the
# model piece by piece, and then rearranged back together.
# 
# While doing this, one of the most important steps is to ensure some
# overlap between each of the chunks, to accomodate for artifacts at the
# edges. This process of chunking and arrangement can be done in various
# ways, but an example implementation can be seen in the next few cells.
# 

from torch.nn import functional as F

# just leave overlap and segmenent, If device is provided it can be useful in some cases when separating very long tracks, cross fading at the 
def apply_model(model, mix, segment=10.,
                overlap=0.25, device=None,
                ):
    """
    Apply model to a given mixture.

    Args:
        segment (int): What the segment length will be in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    kwargs = {
        'segment': segment,
        'overlap': overlap,
        'device': device,
    }

    model.to(device)
    batch, channels, length = mix.shape
    if segment != 0:
        kwargs['segment'] = 0
        out = torch.zeros(batch, 4, channels, length, device=mix.device)
        sum_weight = torch.zeros(length, device=mix.device)
        segment = int(sample_rate * segment)
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = stride / sample_rate
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment.
        weight = torch.cat([torch.arange(1, segment // 2 + 1, device=device),
                         torch.arange(segment - segment // 2, 0, -1, device=device)])
        assert len(weight) == segment
        weight = (weight / weight.max())
        futures = []
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            future = apply_model(model, chunk, **kwargs)
            futures.append((future, offset))
            offset += segment
        for future, offset in futures:
            chunk_out = future
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += (weight[:chunk_length] * chunk_out).to(mix.device)
            sum_weight[offset:offset + segment] += weight[:chunk_length].to(mix.device)
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    else:
        valid_length = length
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(valid_length).to(device)
        with torch.no_grad():
            out = model.forward(padded_mix)
        return center_trim(out, length)

class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, torch.Tensor)
        return TensorChunk(tensor_or_chunk)
    
def center_trim(tensor: torch.Tensor, reference: typing.Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

def plot_spectrogram(stft, title="Spectrogram", xlim=None):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(spectrogram, cmap="viridis", vmin=-100, vmax=0, origin="lower", aspect="auto")
    figure.suptitle(title)
    plt.colorbar(img, ax=axis)
    plt.show()


######################################################################
# 5. Run Model
# ------------
# 
# Finally, we run the model and store the separate source files in a
# directory
# 
# As a test song, we will be using Strand Of Oaks by Spacestation from
# MedleyDB (Creative Commons BY-NC-SA 4.0)
# 
# In order to test with a different song, the variable names and urls
# below can be changed alongside with the parameters to test the song
# separator in different ways.
# 

import requests

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#We download the audio file from our storage. Feel free to download another file and use audio from a specific path
SAMPLE_SONG = download_asset("tutorial-assets/hdemucs_mixture.wav")
waveform, sample_rate = torchaudio.load(SAMPLE_SONG)
store = waveform
# waveform, sample_rate = torchaudio.load(SAMPLE_WAV) #replace SAMPLE_WAV with path

#parameters
segment: int = 10
overlap = 0.25

output_folder = "output"
song_name = "test" #change accordingly
mp3 = False #If track is MP3 file, change to True

print(f"Separating track")

ref = waveform.mean(0)
waveform = (waveform - ref.mean()) / ref.std() #normalization

sources = apply_model(model, waveform[None], device=device, 
                      segment=segment, overlap=overlap,
                      )[0]
sources = sources * ref.std() + ref.mean()

sources_list = model.sources
sources = list(sources)

audios = dict(zip(sources_list, sources))


######################################################################
# 5.1 Separate Track
# ^^^^^^^^^^^^^^^^^^
# 
# The default set of pretrained weights that has been loaded has 4 sources
# that it is separated into: drums, bass, other, and vocals in that order.
# They have been stored into the dict “audios” and therefore can be
# accessed there. For the four sources, there is a separate cell for each,
# that will create the audio, the spectrogram graph, and also calculate
# the sdr score.
# 

N_FFT = 4096
N_HOP = 4
stft = torchaudio.transforms.Spectrogram(
    n_fft=N_FFT,
    hop_length=N_HOP,
    power=None,
)


######################################################################
# 5.2 Original Track
# ^^^^^^^^^^^^^^^^^^
# 
# This is the original track’s audio and spectrogram
# 

track = store
segment = track[:, 10*sample_rate: 20*sample_rate]
plot_spectrogram(stft(segment)[0], "Spectrogram Mixture")
Audio(track, rate=sample_rate)


######################################################################
# 5.3 Drums
# ^^^^^^^^^
# 
# Drums audio, spectrogram graph, and SDR
# 

#DRUMS

track = audios["drums"]
#will take a chunk of 10 seconds in the middle
segment = track[:, 10*sample_rate: 20*sample_rate]
plot_spectrogram(stft(segment)[0], "Spectrogram Drums")

Audio(track, rate=sample_rate)


######################################################################
# 5.4 Bass
# ^^^^^^^^
# 
# Bass audio, spectrogram graph, and SDR
# 

#Bass
track = audios["bass"]
#will take a chunk of 5 seconds in the middle between seconds 20-25
segment = track[:, 20*sample_rate: 25*sample_rate]
plot_spectrogram(stft(segment)[0], "Spectrogram Bass")

Audio(track, rate=sample_rate)


######################################################################
# 5.5 Other
# ^^^^^^^^^
# 
# Other audio, spectrogram graph, and SDR
# 

#Other
track = audios["other"]

#will take a chunk of 5 seconds in the middle between seconds 20-25
segment = track[:, 20*sample_rate: 25*sample_rate]
plot_spectrogram(stft(segment)[0], "Spectrogram Other")

Audio(track, rate=sample_rate)


######################################################################
# 5.6 Vocals
# ^^^^^^^^^^
# 
# Vocals audio, spectrogram graph, and SDR
# 

#Vocals
track = audios["vocals"]

#will take a chunk of 5 seconds in the middle between seconds 20-25
segment = track[:, 20*sample_rate: 25*sample_rate]
plot_spectrogram(stft(segment)[0], "Spectrogram Vocals")

Audio(track, rate=sample_rate)

