torchaudio
==========
This library is part of the `PyTorch
<http://pytorch.org/>`_ project. PyTorch is an open source
machine learning framework.

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


The :mod:`torchaudio` package consists of I/O, popular datasets and common audio transformations.

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

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
   prototype

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   auto_examples/wav2vec2/index
   auto_examples/tts/index

.. toctree::
   :maxdepth: 1
   :caption: PyTorch Libraries

   PyTorch <https://pytorch.org/docs>
   torchaudio <https://pytorch.org/audio>
   torchtext <https://pytorch.org/text>
   torchvision <https://pytorch.org/vision>
   TorchElastic <https://pytorch.org/elastic/>
   TorchServe <https://pytorch.org/serve>
   PyTorch on XLA Devices <http://pytorch.org/xla/>


Citing torchaudio
~~~~~~~~~~~~~~~~~

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
      author={Yao-Yuan Yang and Moto Hira and Zhaoheng Ni and Anjali Chourdia and Artyom Astafurov
              and Caroline Chen and Ching-Feng Yeh and Christian Puhrsch and David Pollack and
              Dmitriy Genzel and Donny Greenberg and Edward Z. Yang and Jason Lian and Jay
              Mahadeokar and Jeff Hwang and Ji Chen and Peter Goldsborough and Prabhat Roy and
              Sean Narenthiran and Shinji Watanabe and Soumith Chintala and Vincent
              Quenneville-Bélair and Yangyang Shi},
      journal={arXiv preprint arXiv:2110.15018},
      year={2021}
    }

