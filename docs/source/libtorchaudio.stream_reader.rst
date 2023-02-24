.. warning::
   TorchAudio's C++ API is a prototype feature.
   API/ABI backward compatibility is not guaranteed.

torchaudio::io::StreamReader
============================

.. doxygenclass:: torchaudio::io::StreamReader

Constructors
------------

.. doxygenfunction:: torchaudio::io::StreamReader::StreamReader(const std::string &src, const c10::optional<std::string> &format = {}, const c10::optional<OptionDict> &option = {})

Query Methods
-------------

find_best_audio_stream
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::find_best_audio_stream

find_best_video_stream
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::find_best_video_stream

get_metadata
^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::get_metadata

num_src_streams
^^^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::num_src_streams

get_src_stream_info
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamReader::get_src_stream_info

num_out_streams
^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamReader::num_out_streams

get_out_stream_info
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamReader::get_out_stream_info

is_buffer_ready
^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamReader::is_buffer_ready

Configure Methods
-----------------

add_audio_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamReader::add_audio_stream

add_video_stream
^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::add_video_stream

remove_stream
^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::remove_stream

Stream Methods
^^^^^^^^^^^^^^

seek
^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::seek

process_packet
^^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::process_packet

process_packet_block
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::process_packet_block

process_all_packets
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::process_all_packets

fill_buffer
^^^^^^^^^^^
.. doxygenfunction:: torchaudio::io::StreamReader::fill_buffer

Retrieval Methods
-----------------

pop_chunks
^^^^^^^^^^

.. doxygenfunction:: torchaudio::io::StreamReader::pop_chunks


Support Structures
------------------

Chunk
^^^^^

.. container:: py attribute

   .. doxygenstruct:: torchaudio::io::Chunk
      :members:

SrcStreaminfo
^^^^^^^^^^^^^

.. container:: py attribute

   .. doxygenstruct:: torchaudio::io::SrcStreamInfo
      :members:

OutputStreaminfo
^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. doxygenstruct:: torchaudio::io::OutputStreamInfo
      :members:
