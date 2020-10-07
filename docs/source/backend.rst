.. _backend:

torchaudio.backend
==================

Overview
~~~~~~~~

:mod:`torchaudio.backend` module provides implementations for audio file I/O functionalities, which are ``torchaudio.info``, ``torchaudio.load``, ``torchaudio.load_wav`` and ``torchaudio.save``.

There are currently four implementations available.

* :ref:`"sox" <sox_backend>` (deprecated, default on Linux/macOS)
* :ref:`"sox_io" <sox_io_backend>` (default on Linux/macOS from the 0.8.0 release)
* :ref:`"soundfile" - legacy interface <soundfile_legacy_backend>` (deprecated, default on Windows)
* :ref:`"soundfile" - new interface <soundfile_backend>` (default on Windows from the 0.8.0 release)

On Windows, only the ``"soundfile"`` backend (with both interfaces) are available. It is recommended to use the new interface as the legacy interface is deprecated.
On Linux/macOS, please use "sox_io" backend. The use of ``"sox"`` backend is strongly discouraged as it cannot correctly handle formats other than 16-bit integer WAV. See `#726 <https://github.com/pytorch/audio/pull/726>`_ for the detail.

.. note::
   Instead of calling functions in ``torchaudio.backend`` directly, please use ``torchaudio.info``, ``torchaudio.load``, ``torchaudio.load_wav`` and ``torchaudio.save`` with proper backend set with :func:`torchaudio.set_audio_backend`.

Availability
------------

``"sox"`` and ``"sox_io"`` backends require C++ extension module. Linux and macOS binary distributions include this. These backends are not available on Windows.

``"soundfile"`` backend requires ``SoundFile``. Please refer to `the SoundFile documentation <https://pysoundfile.readthedocs.io/en/latest/>`_ for the installation.

Changes in default backend and deprecation
------------------------------------------

Backend module is going through a major overhaul. The following table summarizes the timeline for the changes and deprecations.

 +--------------------+--------------------------+-----------------------+------------------------+
 | **Backend**        | **0.7.0**                | **0.8.0**             | **0.9.0**              |
 +====================+==========================+=======================+========================+
 | ``"sox"``          | Default on Linux/macOS   | Available             | Removed                |
 | (deprecated)       |                          |                       |                        |
 +--------------------+--------------------------+-----------------------+------------------------+
 | ``"sox_io"``       | Available                | Default on Linx/macOS | Default on Linux/macOS |
 +--------------------+--------------------------+-----------------------+------------------------+
 | ``"soundfile"``    | Default on Windows       | Available             | Removed                |
 | (legacy interface, |                          |                       |                        |
 | deprecated)        |                          |                       |                        |
 +--------------------+--------------------------+-----------------------+------------------------+
 | ``"soundfile"``    | Available                | Default on Windows    | Default on Windows     |
 | (new interface)    |                          |                       |                        |
 +--------------------+--------------------------+-----------------------+------------------------+

* The default backend for Linux/macOS will be changed from ``"sox"`` to ``"sox_io"`` in ``0.8.0`` release.
* The ``"sox"`` backend will be removed in the ``0.9.0`` release.
* Starting from the 0.8.0 release, ``"soundfile"`` backend will use the new interface, which has the same interface as ``"sox_io"`` backend. The legacy interface will be removed in the ``0.9.0`` release.

Common Data Structure
~~~~~~~~~~~~~~~~~~~~~

Structures used to report the metadata of audio files.

AudioMetaData
-------------

.. autoclass:: torchaudio.backend.common.AudioMetaData

SignalInfo (Deprecated)
-----------------------

.. autoclass:: torchaudio.backend.common.SignalInfo

EncodingInfo (Deprecated)
-------------------------

.. autoclass:: torchaudio.backend.common.EncodingInfo

.. _sox_backend:

Sox Backend (Deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~

The ``"sox"`` backend is available on Linux/macOS and not available on Windows. This backend is currently the default when available, but is deprecated and will be removed in ``0.9.0`` release.

You can switch from another backend to ``sox`` backend with the following;

.. code::

   torchaudio.set_audio_backend("sox")

info
----

.. autofunction:: torchaudio.backend.sox_backend.info

load
----

.. autofunction:: torchaudio.backend.sox_backend.load

.. autofunction:: torchaudio.backend.sox_backend.load_wav


save
----

.. autofunction:: torchaudio.backend.sox_backend.save

others
------

.. automodule:: torchaudio.backend.sox_backend
   :members:
   :exclude-members: info, load, load_wav, save

.. _sox_io_backend:

Sox IO Backend
~~~~~~~~~~~~~~

The ``"sox_io"`` backend is available on Linux/macOS and not available on Windows. This backend is recommended over the ``"sox"`` backend, and will become the default in the ``0.8.0`` release.

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

.. autofunction:: torchaudio.backend.sox_io_backend.load_wav


save
----

.. autofunction:: torchaudio.backend.sox_io_backend.save

.. _soundfile_legacy_backend:

Soundfile Backend
~~~~~~~~~~~~~~~~~

The ``"soundfile"`` backend is available when `SoundFile <https://pysoundfile.readthedocs.io/en/latest/>`_ is installed. This backend is the default on Windows.

The ``"soundfile"`` backend has two interfaces, legacy and new.

* In the ``0.7.0`` release, the legacy interface is used by default when switching to the ``"soundfile"`` backend.
* In the ``0.8.0`` release, the new interface will become the default.
* In the ``0.9.0`` release, the legacy interface will be removed.

To change the interface, set ``torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE`` flag **before** switching the backend.

.. code::

   torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = True
   torchaudio.set_audio_backend("soundfile")  # The legacy interface

   torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
   torchaudio.set_audio_backend("soundfile")  # The new interface

Legacy Interface (Deprecated)
-----------------------------

``"soundfile"`` backend with legacy interface is currently the default on Windows, however this interface is deprecated and will be removed in the ``0.9.0`` release.

To switch to this backend/interface, set ``torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE`` flag **before** switching the backend.

.. code::

   torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = True
   torchaudio.set_audio_backend("soundfile")  # The legacy interface

info
^^^^

.. autofunction:: torchaudio.backend.soundfile_backend.info

load
^^^^

.. autofunction:: torchaudio.backend.soundfile_backend.load

.. autofunction:: torchaudio.backend.soundfile_backend.load_wav


save
^^^^

.. autofunction:: torchaudio.backend.soundfile_backend.save

.. _soundfile_backend:

New Interface
-------------

The ``"soundfile"`` backend with new interface will become the default in the ``0.8.0`` release.

To switch to this backend/interface, set ``torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE`` flag **before** switching the backend.

.. code::

   torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
   torchaudio.set_audio_backend("soundfile")  # The new interface

info
^^^^

.. autofunction:: torchaudio.backend._soundfile_backend.info

load
^^^^

.. autofunction:: torchaudio.backend._soundfile_backend.load

.. autofunction:: torchaudio.backend._soundfile_backend.load_wav


save
^^^^

.. autofunction:: torchaudio.backend._soundfile_backend.save
