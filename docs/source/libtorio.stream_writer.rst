
.. warning::
    Starting with version 2.8, we are refactoring TorchAudio to transition it
    into a maintenance phase. As a result:

    - ``torio`` is deprecated in 2.8 and will be removed in 2.9.
    - The decoding and encoding capabilities of PyTorch for both audio and video
      are being consolidated into TorchCodec.

    Please see https://github.com/pytorch/audio/issues/3902 for more information.

.. note::
   The top-level namespace has been changed from ``torchaudio`` to ``torio``.
   ``StreamWriter`` has been renamed to ``StreamingMediaEncoder``.


torio::io::StreamingMediaEncoder
================================

``StreamingMediaEncoder`` is the implementation used by Python equivalent and provides similar interface.
When working with custom I/O, such as in-memory data, ``StreamingMediaEncoderCustomIO`` class can be used.

Both classes have the same methods defined, so their usages are the same.

Constructors
------------

StreamingMediaEncoder
^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamingMediaEncoder

.. doxygenfunction:: torio::io::StreamingMediaEncoder::StreamingMediaEncoder(const std::string &dst, const std::optional<std::string> &format = {})

StreamingMediaEncoderCustomIO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamingMediaEncoderCustomIO

.. doxygenfunction:: torio::io::StreamingMediaEncoderCustomIO::StreamingMediaEncoderCustomIO

Config methods
--------------

add_audio_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::add_audio_stream

add_video_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::add_video_stream

set_metadata
^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::set_metadata

Write methods
-------------

open
^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::open

close
^^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::close

write_audio_chunk
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::write_audio_chunk

write_video_chunk
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::write_video_chunk

flush
^^^^^

.. doxygenfunction:: torio::io::StreamingMediaEncoder::flush
