torchaudio.prototype
====================

.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result, the ``prototype`` module is
    deprecated in 2.8 and will be removed in 2.9.

``torchaudio.prototype`` provides prototype features;
they are at an early stage for feedback and testing.
Their interfaces might be changed without prior notice.

Most modules of prototypes are excluded from release.
Please refer to `here <https://pytorch.org/audio>`_ for
more information on prototype features.

The modules under ``torchaudio.prototype`` must be
imported explicitly, e.g.

.. code-block:: python

   import torchaudio.prototype.models

.. toctree::
   prototype.datasets
   prototype.functional
   prototype.models
   prototype.pipelines
   prototype.transforms
