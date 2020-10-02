.. _backend:

torchaudio.backend
==================

Overview
~~~~~~~~

:mod:`torchaudio.backend` module provides implementations for audio file I/O, using different backend libraries.
To switch backend, use :py:func:`torchaudio.set_audio_backend`. To check the current backend use :py:func:`torchaudio.get_audio_backend`.

There are currently four implementations available.

    * ``"sox"`` (deprecated)
    * ``"sox_io"``
    * ``"soundfile"``, legacy interface (deprecated)
    * ``"soundfile"``, new interface

``"sox"`` backend is the original backend which is built on ``libsox``. This backend requires C++ extension module and is not available on Windows system. This backend is currently the default backend when available, but is being deprecated. Please use ``"sox_io"`` backend.

``"sox_io"`` backend is the new backend which is built on ``libsox``, and bound with ``TorchScript``. This backend requires C++ extension module and is not available on Windows system. Function calls to this backend are TorchScript-able. This backend will become the default backend in ``"0.8.0"`` release.

``"soundfile"`` backend is built on ``PySoundFile``. You need to install ``PySoundFile`` separately. The current interface of this backend (legacy interface) will be changed in ``0.8.0`` (new interface) to match the ``"sox_io"`` backend.

.. note::
   Instead of calling functions in :mod:`torchaudio.backend` directly, please use ``torchaudio.info``, ``torhcaudio.load``, ``torchaudio.load_wav`` and ``torchaudio.save`` with proper backend set with :func:`torchaudio.get_audio_backend`.

Changes in default backend and Deprecation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   * The default backend for Linux/macOS will be changed from ``"sox"`` to ``"sox_io"`` in ``0.8.0`` release.
   * ``"sox"``  backend will be removed in ``0.9.0`` release.
   * The function signatures of ``"soundfile"`` backends are changed in ``0.8.0`` to match ``"sox_io"`` backend.
   * To opt-in to the new signature of ``"soundfile"`` backend, do ``torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False`` **before** switching to ``"soundfile"``.

   The following table summarizes the timeline for the changes and deprecations.

   +--------------------+--------------------------+--------------------------+--------------------------+
   | **Backend**        | **0.7.0**                | **0.8.0**                | **0.9.0**                |
   +====================+==========================+==========================+==========================+
   | ``"sox"``          | Default if C++ extension | Available                | Removed                  |
   | (deprecated)       | is available             |                          |                          |
   +--------------------+--------------------------+--------------------------+--------------------------+
   | ``"sox_io"``       | Available                | Default if C++ extension | Default if C++ extension |
   |                    |                          | is available             | is available             |
   +--------------------+--------------------------+--------------------------+--------------------------+
   | ``"soundfile"``    | Default if C++ extension | Available                | Removed                  |
   | (legacy interface, | is not available         |                          |                          |
   | deprecated)        |                          |                          |                          |
   +--------------------+--------------------------+--------------------------+--------------------------+
   | ``"soundfile"``    | Available                | Default if C++ extension | Default if C++ extension |
   | (new interface)    |                          | is not available         | is not available         |
   +--------------------+--------------------------+--------------------------+--------------------------+

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

``"sox"`` backend is available on ``torchaudio`` installation with C++ extension. It is currently not available on Windows system.

``"sox"`` backend is currently the default backend when C++ extension is available, however this backend is deprecated and will be removed in ``0.9.0`` release.

In ``0.8.0`` release, :ref:`"sox_io" backend<sox_io_backend>` will become the default backend. Please migrate to :ref:`"sox_io" backend<sox_io_backend>`.

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

``sox_io`` backend is available on ``torchaudio`` installation with C++ extension. It is currently not available on Windows system.

This new backend is recommended over ``sox`` backend, and will become the default backend in ``0.8.0`` release when C++ extension is present.

You can switch from another backend to ``sox_io`` backend with the following;

.. code::

   torchaudio.set_audio_backend("sox_io")

The function calls to this backend are TorchSript-able. You can apply :func:`torch.jit.script` and dump the object toa file, then call it from C++ application.

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

``soundfile`` backend is available when ``PySoundFile`` is installed. This backend works on ``torchaudio`` installation without C++ extension. (i.e. Windows)

``"soundfile"`` backend has two interfaces, legacy and new.

* In ``0.7.0`` release, the legacy interface is used by default when switching to ``"soundfile"`` backend.
* In ``0.8.0`` release, the new interface will become the default.
* In ``0.9.0`` release, the legacy interface will be removed.

To change the interface, set ``torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE`` flag **before** switching the backend.

.. code::

   torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = True
   torchaudio.set_audio_backend("soundfile")  # The legacy interface

   torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
   torchaudio.set_audio_backend("soundfile")  # The new interface

Legacy Interface (Deprecated)
-----------------------------

``"soundfile"`` backend with legacy interface is currently the default backend if C++ is not available, however this interface is deprecated and will be removed in ``0.9.0`` release.

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

``"soundfile"`` backend with new interface will become the default interface in ``0.8.0``.

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
