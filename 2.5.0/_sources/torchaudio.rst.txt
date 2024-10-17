torchaudio
==========

.. currentmodule:: torchaudio

I/O
---

``torchaudio`` top-level module provides the following functions that make
it easy to handle audio data.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/io.rst

   info
   load
   save
   list_audio_backends

.. _backend:

Backend and Dispatcher
----------------------

Decoding and encoding media is highly elaborated process. Therefore, TorchAudio
relies on third party libraries to perform these operations. These third party
libraries are called ``backend``, and currently TorchAudio integrates the
following libraries.

Please refer to `Installation <./installation.html>`__ for how to enable backends.

Conventionally, TorchAudio has had its I/O backend set globally at runtime
based on availability. However, this approach does not allow applications to
use different backends, and it is not well-suited for large codebases.

For these reasons, in v2.0, we introduced a dispatcher, a new mechanism to allow
users to choose a backend for each function call.

When dispatcher mode is enabled, all the I/O functions accept extra keyward argument
``backend``, which specifies the desired backend. If the specified
backend is not available, the function call will fail.

If a backend is not explicitly chosen, the functions will select a backend to use given order of precedence and library availability.

The following table summarizes the backends.

.. list-table::
   :header-rows: 1
   :widths: 8 12 25 60

   * - Priority
     - Backend
     - Supported OS
     - Note
   * - 1
     - FFmpeg
     - Linux, macOS, Windows
     - Use :py:func:`~torchaudio.utils.ffmpeg_utils.get_audio_decoders` and
       :py:func:`~torchaudio.utils.ffmpeg_utils.get_audio_encoders`
       to retrieve the supported codecs.

       This backend Supports various protocols, such as HTTPS and MP4, and file-like objects.
   * - 2
     - SoX
     - Linux, macOS
     - Use :py:func:`~torchaudio.utils.sox_utils.list_read_formats` and
       :py:func:`~torchaudio.utils.sox_utils.list_write_formats`
       to retrieve the supported codecs.

       This backend does *not* support file-like objects.
   * - 3
     - SoundFile
     - Linux, macOS, Windows
     - Please refer to `the official document <https://pysoundfile.readthedocs.io/>`__ for the supported codecs.

       This backend supports file-like objects.

.. _dispatcher_migration:

Dispatcher Migration
~~~~~~~~~~~~~~~~~~~~

We are migrating the I/O functions to use the dispatcher mechanism, and this
incurs multiple changes, some of which involve backward-compatibility-breaking
changes, and require users to change their function call.

The (planned) changes are as follows. For up-to-date information,
please refer to https://github.com/pytorch/audio/issues/2950

* In 2.0, audio I/O backend dispatcher was introduced.
  Users can opt-in to using dispatcher by setting the environment variable
  ``TORCHAUDIO_USE_BACKEND_DISPATCHER=1``.
* In 2.1, the disptcher became the default mechanism for I/O.
* In 2.2, the legacy global backend mechanism is removed.
  Utility functions :py:func:`get_audio_backend` and :py:func:`set_audio_backend`
  became no-op.

Furthermore, we removed file-like object support from libsox backend, as this
is better supported by FFmpeg backend and makes the build process simpler.
Therefore, beginning with 2.1, FFmpeg and Soundfile are the sole backends that support
file-like objects.
