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
# `MUSDB18-HQ <https://zenodo.org/record/3338373>`__ and other alternative
# sources.
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
from torchaudio.transforms import Fade

# just leave overlap and segmenent, If device is provided it can be useful in some cases when separating very long tracks, cross fading at the 
def apply_model(model, mix, segment=10.,
                overlap=0.1, device=None,
                ):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment. 

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

    model.to(device)
    batch, channels, length = mix.shape
    
    chunk_len = int(sample_rate * segment * (1 + overlap))
    prev = 0
    index = prev + chunk_len
    fade = Fade(fade_in_len = 0, fade_out_len = int(overlap * sample_rate), fade_shape='linear')
    
    final = torch.zeros(batch, 4, channels, length, device=device)
    
    print(prev, index)
    while index < length:
        chunk = mix[:,:,prev:index]
        with torch.no_grad():
            out = model.forward(chunk)
            
        if prev != 0:
            fade = Fade(fade_in_len = int(overlap * sample_rate), fade_out_len = int(overlap * sample_rate), fade_shape='linear')
        out = fade(out)
        final[:,:,:,prev:index] += out
        
        prev = int(index - overlap * sample_rate)
        index += chunk_len
    chunk = mix[:,:,prev:]
    with torch.no_grad():
        out = model.forward(chunk)
    fade = Fade(fade_in_len = int(overlap * sample_rate), fade_out_len = 0, fade_shape='linear')
    out = fade(out)
    final[:,:,:,prev:] += out
    return final

def plot_spectrogram(stft, title="Spectrogram"):
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
overlap = 0.1

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
#will take a chunk of 10 seconds in the middle between seconds 10-20
segment = track[:, 10*sample_rate: 20*sample_rate]

#get original drums track
drums_original = download_asset("tutorial-assets/hdemucs_drums.wav")
waveform, sample_rate = torchaudio.load(drums_original)

#calculate sdr score and print
print("SDR score is:", mir_eval.separation.bss_eval_sources(waveform.detach().numpy(), segment.detach().numpy())[0].mean())

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
#will take a chunk of 10 seconds in the middle between seconds 10-20
segment = track[:, 10*sample_rate: 20*sample_rate]
plot_spectrogram(stft(segment)[0], "Spectrogram Bass")

#get original bass track
bass_original = download_asset("tutorial-assets/hdemucs_bass.wav")
waveform, sample_rate = torchaudio.load(bass_original)

#calculate sdr score and print
print("SDR score is:", mir_eval.separation.bss_eval_sources(waveform.detach().numpy(), segment.detach().numpy())[0].mean())

Audio(track, rate=sample_rate)


######################################################################
# 5.5 Other
# ^^^^^^^^^
# 
# Other audio, spectrogram graph, and SDR
# 

#Other
track = audios["other"]

#will take a chunk of 10 seconds in the middle between seconds 10-20
segment = track[:, 10*sample_rate: 20*sample_rate]
plot_spectrogram(stft(segment)[0], "Spectrogram Other")

#get original other track
other_original = download_asset("tutorial-assets/hdemucs_other.wav")
waveform, sample_rate = torchaudio.load(other_original)

#calculate sdr score and print
print("SDR score is:", mir_eval.separation.bss_eval_sources(waveform.detach().numpy(), segment.detach().numpy())[0].mean())

Audio(track, rate=sample_rate)


######################################################################
# 5.6 Vocals
# ^^^^^^^^^^
# 
# Vocals audio, spectrogram graph, and SDR
# 

#Vocals
track = audios["vocals"]

#will take a chunk of 10 seconds in the middle between seconds 10-20
segment = track[:, 10*sample_rate: 20*sample_rate]

#get original bass track
bass_original = download_asset("tutorial-assets/hdemucs_bass.wav")
waveform, sample_rate = torchaudio.load(bass_original)

#calculate sdr score and print
print("SDR score is:", mir_eval.separation.bss_eval_sources(waveform.detach().numpy(), segment.detach().numpy())[0].mean())

plot_spectrogram(stft(segment)[0], "Spectrogram Vocals")

Audio(track, rate=sample_rate)
