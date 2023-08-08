.. _backend:

torchaudio.backend
==================

.. py:module:: torchaudio.backend

Overview
~~~~~~~~

:mod:`torchaudio.backend` module provides implementations for audio file I/O functionalities, which are ``torchaudio.info``, ``torchaudio.load``, and ``torchaudio.save``.

.. note::
   Release 2.1 will revise ``torchaudio.info``, ``torchaudio.load``, and ``torchaudio.save`` to allow for backend selection via function parameter rather than ``torchaudio.set_audio_backend``, with FFmpeg being the default backend.
   The new logic can be enabled in the current release by setting environment variable ``TORCHAUDIO_USE_BACKEND_DISPATCHER=1``.
   See :ref:`future_api` for details on the new API.

There are currently two implementations available.

* :py:mod:`"sox_io" <torchaudio.backends.sox_io_backend>` (default on Linux/macOS)
* :py:mod:`"soundfile" <torchaudio.backends.soundfile_backend>` (default on Windows)

.. note::
   Instead of calling functions in ``torchaudio.backend`` directly, please use ``torchaudio.info``, ``torchaudio.load``, and ``torchaudio.save`` with proper backend set with :func:`torchaudio.set_audio_backend`.

Availability
------------

``"sox_io"`` backend requires C++ extension module. torchaudio<2.1.0 will include libsox in the Linux/macOS binary distributions, for later versions please install `libsox`. This backend is not available on Windows.

``"soundfile"`` backend requires ``SoundFile``. Please refer to `the SoundFile documentation <https://pysoundfile.readthedocs.io/en/latest/>`_ for the installation.

Common Data Structure
~~~~~~~~~~~~~~~~~~~~~

Structures used to report the metadata of audio files.

AudioMetaData
-------------

.. autoclass:: torchaudio.backend.common.AudioMetaData

.. py:module:: torchaudio.backend.sox_io_backend

Sox IO Backend
~~~~~~~~~~~~~~

The ``sox_io`` backend is available and default on Linux/macOS and not available on Windows.

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

.. py:module:: torchaudio.backend.soundfile_backend

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
