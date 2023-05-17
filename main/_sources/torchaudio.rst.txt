torchaudio
==========

I/O
---

``torchaudio`` top-level module provides the following functions that make
it easy to handle audio data.

- :py:func:`torchaudio.info`
- :py:func:`torchaudio.load`
- :py:func:`torchaudio.save`

Under the hood, these functions are implemented using various decoding/encoding
libraries. There are currently three variants.

- ``FFmpeg``
- ``libsox``
- ``SoundFile``

``libsox`` backend is the first backend implemented in TorchAudio, and it
works on Linux and macOS.
``SoundFile`` backend was added to extend audio I/O support to Windows.
It also works on Linux and macOS.
``FFmpeg`` backend is the latest addition and it supports wide range of audio, video
formats and protocols.
It works on Linux, macOS and Windows.

.. _dispatcher_migration:

Introduction of Dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~~

Conventionally, torchaudio has had its IO backend set globally at runtime based on availability.
However, this approach does not allow applications to use different
backends, and it is not well-suited for large codebases.

For these reasons, we are introducing a dispatcher, a new mechanism to allow users to
choose a backend for each function call, and migrating the I/O functions.
This incurs multiple changes, some of which involve backward-compatibility-breaking changes, and require
users to change their function call.

The (planned) changes are as follows. For up-to-date information,
please refer to https://github.com/pytorch/audio/issues/2950

* In 2.0, audio I/O backend dispatcher was introduced.
  Users can opt-in to using dispatcher by setting the environment variable
  ``TORCHAUDIO_USE_BACKEND_DISPATCHER=1``
* In 2.1, the disptcher becomes the default mechanism for I/O.
  Those who need to keep using the previous mechanism (global backend) can do
  so by setting ``TORCHAUDIO_USE_BACKEND_DISPATCHER=0``.

Furthermore, we are removing file-like object support from libsox backend, as this
is better supported by FFmpeg backend and makes the build process simpler.
Therefore, beginning with 2.1, FFmpeg and Soundfile are the sole backends that support file-like objects.

The changes in 2.1 will mark the :ref:`backend utilities <backend_utils>` deprecated.

Current API
-----------

I/O functionalities
~~~~~~~~~~~~~~~~~~~

Audio I/O functions are implemented in :ref:`torchaudio.backend<backend>` module, but for the ease of use, the following functions are made available on :mod:`torchaudio` module. There are different backends available and you can switch backends with :func:`set_audio_backend`.


Please refer to :ref:`backend` for the detail, and the :doc:`Audio I/O tutorial <../tutorials/audio_io_tutorial>` for the usage.


torchaudio.info
~~~~~~~~~~~~~~~

.. function:: torchaudio.info(filepath: str, ...)

   Fetch meta data of an audio file. Refer to :ref:`backend` for the detail.

torchaudio.load
~~~~~~~~~~~~~~~

.. function:: torchaudio.load(filepath: str, ...)

   Load audio file into torch.Tensor object. Refer to :ref:`backend` for the detail.

torchaudio.save
~~~~~~~~~~~~~~~

.. function:: torchaudio.save(filepath: str, src: torch.Tensor, sample_rate: int, ...)

   Save torch.Tensor object into an audio format. Refer to :ref:`backend` for the detail.

.. currentmodule:: torchaudio

.. _backend_utils:

Backend Utilities
~~~~~~~~~~~~~~~~~

The following functions are effective only when backend dispatcher is disabled.
They are effectively deprecated.

.. autofunction:: list_audio_backends

.. autofunction:: get_audio_backend

.. autofunction:: set_audio_backend

.. _future_api:

Future API
----------

Dispatcher
~~~~~~~~~~

The dispatcher tries to use the I/O backend in the following order of precedence

1. FFmpeg
2. libsox
3. soundfile

One can pass ``backend`` argument to I/O functions to override this.

See :ref:`future_api` for details on the new API.

In the next release, each of ``torchaudio.info``, ``torchaudio.load``, and ``torchaudio.save`` will allow for selecting a backend to use via parameter ``backend``.
The functions will support using any of FFmpeg, SoX, and SoundFile, provided that the corresponding library is installed.
If a backend is not explicitly chosen, the functions will select a backend to use given order of precedence (FFmpeg, SoX, SoundFile) and library availability.

Note that only FFmpeg and SoundFile will support file-like objects.

These functions can be enabled in the current release by setting environment variable ``TORCHAUDIO_USE_BACKEND_DISPATCHER=1``.

.. currentmodule:: torchaudio._backend

torchaudio.info
~~~~~~~~~~~~~~~

.. autofunction:: info
   :noindex:

torchaudio.load
~~~~~~~~~~~~~~~

.. autofunction:: load
   :noindex:

torchaudio.save
~~~~~~~~~~~~~~~

.. autofunction:: save
   :noindex:
