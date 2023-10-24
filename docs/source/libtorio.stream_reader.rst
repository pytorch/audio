.. warning::
   TorchAudio's C++ API is a prototype feature.
   API/ABI backward compatibility is not guaranteed.


.. note::
   The top-level namespace has been changed from ``torchaudio`` to ``torio``.


torio::io::StreamReader
=======================

``StreamReader`` is the implementation used by Python equivalent and provides similar interface.
When working with custom I/O, such as in-memory data, ``StreamReaderCustomIO`` class can be used.

Both classes have the same methods defined, so their usages are the same.

Constructors
------------

StreamReader
^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamReader

.. doxygenfunction:: torio::io::StreamReader::StreamReader(const std::string &src, const c10::optional<std::string> &format = {}, const c10::optional<OptionDict> &option = {})

StreamReaderCustomIO
^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamReaderCustomIO

.. doxygenfunction:: torio::io::StreamReaderCustomIO::StreamReaderCustomIO

Query Methods
-------------

find_best_audio_stream
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::find_best_audio_stream

find_best_video_stream
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::find_best_video_stream

get_metadata
^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::get_metadata

num_src_streams
^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::num_src_streams

get_src_stream_info
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamReader::get_src_stream_info

num_out_streams
^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamReader::num_out_streams

get_out_stream_info
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamReader::get_out_stream_info

is_buffer_ready
^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamReader::is_buffer_ready

Configure Methods
-----------------

add_audio_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamReader::add_audio_stream

add_video_stream
^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::add_video_stream

remove_stream
^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::remove_stream

Stream Methods
^^^^^^^^^^^^^^

seek
^^^^
.. doxygenfunction:: torio::io::StreamReader::seek

process_packet
^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::process_packet()

process_packet_block
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::process_packet_block

process_all_packets
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::process_all_packets

fill_buffer
^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamReader::fill_buffer

Retrieval Methods
-----------------

pop_chunks
^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamReader::pop_chunks


Support Structures
------------------

Chunk
^^^^^

.. container:: py attribute

   .. doxygenstruct:: torio::io::Chunk
      :members:

SrcStreaminfo
^^^^^^^^^^^^^

.. container:: py attribute

   .. doxygenstruct:: torio::io::SrcStreamInfo
      :members:

OutputStreaminfo
^^^^^^^^^^^^^^^^

.. container:: py attribute

   .. doxygenstruct:: torio::io::OutputStreamInfo
      :members:
