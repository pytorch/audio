Torchaudio Documentation
========================

Torchaudio is a library for audio and signal processing with PyTorch.
It provides I/O, signal and data processing functions, datasets,
model implementations and application components.

Features described in this documentation are classified by release status:

  *Stable:*  These features will be maintained long-term and there should generally
  be no major performance limitations or gaps in documentation.
  We also expect to maintain backwards compatibility (although
  breaking changes can happen and notice will be given one release ahead
  of time).

  *Beta:*  Features are tagged as Beta because the API may change based on
  user feedback, because the performance needs to improve, or because
  coverage across operators is not yet complete. For Beta features, we are
  committing to seeing the feature through to the Stable classification.
  We are not, however, committing to backwards compatibility.

  *Prototype:*  These features are typically not available as part of
  binary distributions like PyPI or Conda, except sometimes behind run-time
  flags, and are at an early stage for feedback and testing.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Torchaudio Documentation

   Index <self>

API References
--------------

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   torchaudio
   backend
   functional
   transforms
   datasets
   models
   pipelines
   sox_effects
   compliance.kaldi
   kaldi_io
   utils

Prototype API References
------------------------

.. toctree::
   :maxdepth: 1
   :caption: Prototype API Reference

   prototype
   prototype.ctc_decoder
   prototype.models
   prototype.pipelines

Getting Started
---------------
    
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/audio_io_tutorial
   tutorials/audio_resampling_tutorial
   tutorials/audio_data_augmentation_tutorial
   tutorials/audio_feature_extractions_tutorial
   tutorials/audio_feature_augmentation_tutorial
   tutorials/audio_datasets_tutorial

Advanced Usages
---------------

.. toctree::
   :maxdepth: 1
   :caption: Advanced Usages

   tutorials/speech_recognition_pipeline_tutorial
   tutorials/forced_alignment_tutorial
   tutorials/tacotron2_pipeline_tutorial
   tutorials/mvdr_tutorial
   tutorials/asr_inference_with_ctc_decoder_tutorial

Citing torchaudio
-----------------

If you find torchaudio useful, please cite the following paper:

- Yang, Y.-Y., Hira, M., Ni, Z., Chourdia, A., Astafurov, A., Chen, C., Yeh, C.-F., Puhrsch, C.,
  Pollack, D., Genzel, D., Greenberg, D., Yang, E. Z., Lian, J., Mahadeokar, J., Hwang, J.,
  Chen, J., Goldsborough, P., Roy, P., Narenthiran, S., Watanabe, S., Chintala, S.,
  Quenneville-Bélair, V, & Shi, Y. (2021).
  TorchAudio: Building Blocks for Audio and Speech Processing. arXiv preprint arXiv:2110.15018.


In BibTeX format:

.. code-block:: bibtex

    @article{yang2021torchaudio,
      title={TorchAudio: Building Blocks for Audio and Speech Processing},
      author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and
              Anjali Chourdia and Artyom Astafurov and Caroline Chen and
              Ching-Feng Yeh and Christian Puhrsch and David Pollack and
              Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and
              Jason Lian and Jay Mahadeokar and Jeff Hwang and Ji Chen and
              Peter Goldsborough and Prabhat Roy and Sean Narenthiran and
              Shinji Watanabe and Soumith Chintala and
              Vincent Quenneville-Bélair and Yangyang Shi},
      journal={arXiv preprint arXiv:2110.15018},
      year={2021}
    }

.. toctree::
   :maxdepth: 1
   :caption: PyTorch Libraries
   :hidden:

   PyTorch <https://pytorch.org/docs>
   torchaudio <https://pytorch.org/audio>
   torchtext <https://pytorch.org/text>
   torchvision <https://pytorch.org/vision>
   TorchElastic <https://pytorch.org/elastic/>
   TorchServe <https://pytorch.org/serve>
   PyTorch on XLA Devices <http://pytorch.org/xla/>
