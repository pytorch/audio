.. _backend:

torchaudio.backend
==================

:mod:`torchaudio.backend` module provides implementations for audio file I/O, using different backend libraries.
To switch backend, use :py:func:`torchaudio.set_audio_backend`. To check the current backend use :py:func:`torchaudio.get_audio_backend`.

.. warning::
   Although ``sox`` backend is default for backward compatibility reason, it has a number of issues, therefore it is highly recommended to use ``sox_io`` backend instead. Note, however, that due to the interface refinement, functions defined in ``sox`` backend and those defined in ``sox_io`` backend do not have the same signatures.

.. note::
   Instead of calling functions in :mod:`torchaudio.backend` directly, please use ``torchaudio.info``, ``torhcaudio.load``, ``torchaudio.load_wav`` and ``torchaudio.save`` with proper backend set with :func:`torchaudio.get_audio_backend`.

There are currently three implementations available.

    * :ref:`sox<sox_backend>`
    * :ref:`sox_io<sox_io_backend>`
    * :ref:`soundfile<soundfile_backend>`

``sox`` backend is the original backend which is built on ``libsox``. This module is currently default but is known to have number of issues, such as wrong handling of WAV files other than 16-bit signed integer. Users are encouraged to use ``sox_io`` backend. This backend requires C++ extension module and is not available on Windows system.

``sox_io`` backend is the new backend which is built on ``libsox`` and bound to Python with ``Torchscript``. This module addresses all the known issues ``sox`` backend has. Function calls to this backend can be Torchscriptable. This backend requires C++ extension module and is not available on Windows system.

``soundfile`` backend is built on ``PySoundFile``. You need to install ``PySoundFile`` separately.

Common Data Structure
~~~~~~~~~~~~~~~~~~~~~

Structures used to exchange data between Python interface and ``libsox``. They are used by :ref:`sox<sox_backend>` and :ref:`soundfile<soundfile_backend>` but not by :ref:`sox_io<sox_io_backend>`.

.. autoclass:: torchaudio.backend.common.SignalInfo

.. autoclass:: torchaudio.backend.common.EncodingInfo

.. _sox_backend:

Sox Backend
~~~~~~~~~~~

``sox`` backend is available on ``torchaudio`` installation with C++ extension. It is currently not available on Windows system.

It is currently default backend when it's available. You can switch from another backend to ``sox`` backend with the following;

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

This new backend is recommended over ``sox`` backend. You can switch from another backend to ``sox_io`` backend with the following;

.. code::

   torchaudio.set_audio_backend("sox_io")

The function call to this backend can be Torchsript-able. You can apply :func:`torch.jit.script` and dump the object to file, then call it from C++ application.

info
----

.. autoclass:: torchaudio.backend.sox_io_backend.AudioMetaData

.. autofunction:: torchaudio.backend.sox_io_backend.info

load
----

.. autofunction:: torchaudio.backend.sox_io_backend.load

.. autofunction:: torchaudio.backend.sox_io_backend.load_wav


save
----

.. autofunction:: torchaudio.backend.sox_io_backend.save

.. _soundfile_backend:

Soundfile Backend
~~~~~~~~~~~~~~~~~

``soundfile`` backend is available when ``PySoundFile`` is installed. This backend works on ``torchaudio`` installation without C++ extension. (i.e. Windows)

You can switch from another backend to ``soundfile`` backend with the following;

.. code::

   torchaudio.set_audio_backend("soundfile")

info
----

.. autofunction:: torchaudio.backend.soundfile_backend.info

load
----

.. autofunction:: torchaudio.backend.soundfile_backend.load

.. autofunction:: torchaudio.backend.soundfile_backend.load_wav


save
----

.. autofunction:: torchaudio.backend.soundfile_backend.save
