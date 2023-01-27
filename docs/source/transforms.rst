.. py:module:: torchaudio.transforms

torchaudio.transforms
=====================

.. currentmodule:: torchaudio.transforms

``torchaudio.transforms`` module contains common audio processings and feature extractions. The following diagram shows the relationship between some of the available transforms.


.. image:: https://download.pytorch.org/torchaudio/tutorial-assets/torchaudio_feature_extractions.png

Transforms are implemented using :class:`torch.nn.Module`. Common ways to build a processing pipeline are to define custom Module class or chain Modules together using :class:`torch.nn.Sequential`, then move it to a target device and data type.

.. code::

   # Define custom feature extraction pipeline.
   #
   # 1. Resample audio
   # 2. Convert to power spectrogram
   # 3. Apply augmentations
   # 4. Convert to mel-scale
   #
   class MyPipeline(torch.nn.Module):
       def __init__(
           self,
           input_freq=16000,
           resample_freq=8000,
           n_fft=1024,
           n_mel=256,
           stretch_factor=0.8,
       ):
           super().__init__()
           self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

           self.spec = Spectrogram(n_fft=n_fft, power=2)

           self.spec_aug = torch.nn.Sequential(
               TimeStretch(stretch_factor, fixed_rate=True),
               FrequencyMasking(freq_mask_param=80),
               TimeMasking(time_mask_param=80),
           )

           self.mel_scale = MelScale(
               n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

       def forward(self, waveform: torch.Tensor) -> torch.Tensor:
           # Resample the input
           resampled = self.resample(waveform)

           # Convert to power spectrogram
           spec = self.spec(resampled)

           # Apply SpecAugment
           spec = self.spec_aug(spec)

           # Convert to mel-scale
           mel = self.mel_scale(spec)

           return mel


.. code::

   # Instantiate a pipeline
   pipeline = MyPipeline()

   # Move the computation graph to CUDA
   pipeline.to(device=torch.device("cuda"), dtype=torch.float32)

   # Perform the transform
   features = pipeline(waveform)

Please check out tutorials that cover in-depth usage of trasforms.

.. minigallery:: torchaudio.transforms

Utility
-------

.. autosummary::
    :toctree: generated
    :nosignatures:

    AmplitudeToDB
    MuLawEncoding
    MuLawDecoding
    Resample
    Fade
    Vol
    Loudness
    AddNoise
    Convolve
    FFTConvolve
    Speed
    SpeedPerturbation
    Deemphasis
    Preemphasis

Feature Extractions
-------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    Spectrogram
    InverseSpectrogram
    MelScale
    InverseMelScale
    MelSpectrogram
    GriffinLim
    MFCC
    LFCC
    ComputeDeltas
    PitchShift
    SlidingWindowCmn
    SpectralCentroid
    Vad

Augmentations
-------------

The following transforms implement popular augmentation techniques known as *SpecAugment* :cite:`specaugment`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    FrequencyMasking
    TimeMasking
    TimeStretch

Loss
----

.. autosummary::
    :toctree: generated
    :nosignatures:

    RNNTLoss

Multi-channel
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    PSD
    MVDR
    RTFMVDR
    SoudenMVDR
