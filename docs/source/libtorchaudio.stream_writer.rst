.. warning::
   TorchAudio's C++ API is prototype feature.
   API/ABI backward compatibility is not guaranteed.

torchaudio::io::StreamWriter
============================

.. doxygenclass:: torchaudio::io::StreamWriter

Constructors
------------

.. doxygenfunction:: torchaudio::io::StreamWriter::StreamWriter(const std::string &dst, const c10::optional<std::string> &format = {})

Config methods
--------------

add_audio_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::add_audio_stream

add_video_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::add_video_stream

set_metadata
^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::set_metadata

Write methods
-------------

open
^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::open

close
^^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::close

write_audio_chunk
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::write_audio_chunk

write_video_chunk
^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::write_video_chunk

flush
^^^^^

.. doxygenfunction:: torchaudio::io::StreamWriter::flush
