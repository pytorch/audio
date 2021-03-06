.. _backend:

torchaudio.backend
==================

Overview
~~~~~~~~

:mod:`torchaudio.backend` module provides implementations for audio file I/O functionalities, which are ``torchaudio.info``, ``torchaudio.load``, and ``torchaudio.save``.

There are currently four implementations available.

* :ref:`"sox_io" <sox_io_backend>` (default on Linux/macOS)
* :ref:`"soundfile" <soundfile_backend>` (default on Windows)

.. note::
   Instead of calling functions in ``torchaudio.backend`` directly, please use ``torchaudio.info``, ``torchaudio.load``, and ``torchaudio.save`` with proper backend set with :func:`torchaudio.set_audio_backend`.

Availability
------------

``"sox_io"`` backend requires C++ extension module, which is included in Linux/macOS binary distributions. This backend is not available on Windows.

``"soundfile"`` backend requires ``SoundFile``. Please refer to `the SoundFile documentation <https://pysoundfile.readthedocs.io/en/latest/>`_ for the installation.

Common Data Structure
~~~~~~~~~~~~~~~~~~~~~

Structures used to report the metadata of audio files.

AudioMetaData
-------------

.. autoclass:: torchaudio.backend.common.AudioMetaData

.. _sox_io_backend:

Sox IO Backend
~~~~~~~~~~~~~~

The ``"sox_io"`` backend is available and default on Linux/macOS and not available on Windows.

I/O functions of this backend support `TorchScript <https://pytorch.org/docs/stable/jit.html>`_.

You can switch from another backend to the ``sox_io`` backend with the following;

.. code::

   torchaudio.set_audio_backend("sox_io")

info
----

.. autofunction:: torchaudio.backend.sox_io_backend.info

load
----

.. autofunction:: torchaudio.backend.sox_io_backend.load

save
----

.. autofunction:: torchaudio.backend.sox_io_backend.save

.. _soundfile_backend:

Soundfile Backend
~~~~~~~~~~~~~~~~~

The ``"soundfile"`` backend is available when `SoundFile <https://pysoundfile.readthedocs.io/en/latest/>`_ is installed. This backend is the default on Windows.

You can switch from another backend to the ``"soundfile"`` backend with the following;

.. code::

   torchaudio.set_audio_backend("soundfile")

info
----

.. autofunction:: torchaudio.backend.soundfile_backend.info

load
----

.. autofunction:: torchaudio.backend.soundfile_backend.load

save
----

.. autofunction:: torchaudio.backend.soundfile_backend.save
