.. warning::
   TorchAudio's C++ API is prototype feature.
   API/ABI backward compatibility is not guaranteed.


.. note::
   The top-level namespace has been changed from ``torchaudio`` to ``torio``.


torio::io::StreamWriter
=======================

``StreamWriter`` is the implementation used by Python equivalent and provides similar interface.
When working with custom I/O, such as in-memory data, ``StreamWriterCustomIO`` class can be used.

Both classes have the same methods defined, so their usages are the same.

Constructors
------------

StreamWriter
^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamWriter

.. doxygenfunction:: torio::io::StreamWriter::StreamWriter(const std::string &dst, const c10::optional<std::string> &format = {})

StreamWriterCustomIO
^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamWriterCustomIO

.. doxygenfunction:: torio::io::StreamWriterCustomIO::StreamWriterCustomIO

Config methods
--------------

add_audio_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamWriter::add_audio_stream

add_video_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamWriter::add_video_stream

set_metadata
^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamWriter::set_metadata

Write methods
-------------

open
^^^^

.. doxygenfunction:: torio::io::StreamWriter::open

close
^^^^^

.. doxygenfunction:: torio::io::StreamWriter::close

write_audio_chunk
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamWriter::write_audio_chunk

write_video_chunk
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamWriter::write_video_chunk

flush
^^^^^

.. doxygenfunction:: torio::io::StreamWriter::flush
