torchaudio.prototype
====================

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
    prototype.ctc_decoder
    prototype.models
    prototype.pipelines
