torchaudio.prototype
====================

.. warning::
    As TorchAudio is no longer being actively developed, this functionality will no longer be supported.
    See https://github.com/pytorch/audio/issues/3902 for more details.

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
