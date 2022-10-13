Manipulating audio tensors
==========================

``torchaudio`` implements various operations used in audio-domain.

The following tutorials show how to manipulate audio tensors (waveforms).

Resampling
----------

The resampling operations take advantage of PyTorch's fast convolution
operation, and they can be executed on GPU as-well.

The following tutorial goes into details of
how to use and what parameter affects what perspective of quality.

.. toctree::
   :maxdepth: 1

   tutorials/audio_resampling_tutorial


Augmentations and feature extractions
-------------------------------------

The following tutorials explain various techniques useful for building
preprocessing pipelines.

.. toctree::
   :maxdepth: 1

   tutorials/audio_data_augmentation_tutorial
   tutorials/audio_feature_extractions_tutorial
   tutorials/audio_feature_augmentation_tutorial

Datasets
--------

.. toctree::
   :maxdepth: 1

   tutorials/audio_datasets_tutorial
