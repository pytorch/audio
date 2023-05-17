torchaudio
==========

I/O functionalities
~~~~~~~~~~~~~~~~~~~

Audio I/O functions are implemented in :ref:`torchaudio.backend<backend>` module, but for the ease of use, the following functions are made available on :mod:`torchaudio` module. There are different backends available and you can switch backends with :func:`set_audio_backend`.

Refer to :ref:`backend` for the detail.

.. function:: torchaudio.info(filepath: str, ...)

   Fetch meta data of an audio file. Refer to :ref:`backend` for the detail.

.. function:: torchaudio.load(filepath: str, ...)

   Load audio file into torch.Tensor object. Refer to :ref:`backend` for the detail.

.. function:: torchaudio.save(filepath: str, src: torch.Tensor, sample_rate: int, ...)

   Save torch.Tensor object into an audio format. Refer to :ref:`backend` for the detail.

.. currentmodule:: torchaudio

Backend Utilities
~~~~~~~~~~~~~~~~~

.. autofunction:: list_audio_backends

.. autofunction:: get_audio_backend

.. autofunction:: set_audio_backend
