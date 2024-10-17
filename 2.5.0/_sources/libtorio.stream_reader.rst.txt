.. warning::
   TorchAudio's C++ API is a prototype feature.
   API/ABI backward compatibility is not guaranteed.


.. note::
   The top-level namespace has been changed from ``torchaudio`` to ``torio``.
   ``StreamReader`` has been renamed to ``StreamingMediaDecoder``.


torio::io::StreamingMediaDecoder
================================

``StreamingMediaDecoder`` is the implementation used by Python equivalent and provides similar interface.
When working with custom I/O, such as in-memory data, ``StreamingMediaDecoderCustomIO`` class can be used.

Both classes have the same methods defined, so their usages are the same.

Constructors
------------

StreamingMediaDecoder
^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamingMediaDecoder

.. doxygenfunction:: torio::io::StreamingMediaDecoder::StreamingMediaDecoder(const std::string &src, const std::optional<std::string> &format = {}, const c10::optional<OptionDict> &option = {})

StreamingMediaDecoderCustomIO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torio::io::StreamingMediaDecoderCustomIO

.. doxygenfunction:: torio::io::StreamingMediaDecoderCustomIO::StreamingMediaDecoderCustomIO

Query Methods
-------------

find_best_audio_stream
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::find_best_audio_stream

find_best_video_stream
^^^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::find_best_video_stream

get_metadata
^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::get_metadata

num_src_streams
^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::num_src_streams

get_src_stream_info
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaDecoder::get_src_stream_info

num_out_streams
^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaDecoder::num_out_streams

get_out_stream_info
^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaDecoder::get_out_stream_info

is_buffer_ready
^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaDecoder::is_buffer_ready

Configure Methods
-----------------

add_audio_stream
^^^^^^^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaDecoder::add_audio_stream

add_video_stream
^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::add_video_stream

remove_stream
^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::remove_stream

Stream Methods
^^^^^^^^^^^^^^

seek
^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::seek

process_packet
^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::process_packet()

process_packet_block
^^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::process_packet_block

process_all_packets
^^^^^^^^^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::process_all_packets

fill_buffer
^^^^^^^^^^^
.. doxygenfunction:: torio::io::StreamingMediaDecoder::fill_buffer

Retrieval Methods
-----------------

pop_chunks
^^^^^^^^^^

.. doxygenfunction:: torio::io::StreamingMediaDecoder::pop_chunks


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
