.. warning::
   TorchAudio's C++ API is prototype feature.
   API/ABI backward compatibility is not guaranteed.


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
