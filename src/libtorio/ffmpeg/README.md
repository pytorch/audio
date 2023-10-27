# FFMpeg binding dev note

The ffmpeg binding is based on ver 4.1.

## Learning material

For understanding the concept of stream processing, some tutorials are useful.

https://github.com/leandromoreira/ffmpeg-libav-tutorial

The best way to learn how to use ffmpeg is to look at the official examples.
Practically all the code is re-organization of examples;

https://ffmpeg.org/doxygen/4.1/examples.html

## StreamingMediaDecoder Architecture

The top level class is `StreamingMediaDecoder` class. This class handles the input (via `AVFormatContext*`), and manages `StreamProcessor`s for each stream in the input.

The `StreamingMediaDecoder` object slices the input data into a series of `AVPacket` objects and it feeds the objects to corresponding `StreamProcessor`s.

```
 StreamingMediaDecoder
┌─────────────────────────────────────────────────┐
│                                                 │
│ AVFormatContext*       ┌──► StreamProcessor[0]  │
│          │             │                        │
│          └─────────────┼──► StreamProcessor[1]  │
│      AVPacket*         │                        │
│                        └──► ...                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

The `StreamProcessor` class is composed of one `Decoder` and multiple of `Sink` objects.

`Sink` objects correspond to output streams that users set.
`Sink` class is a wrapper `FilterGraph` and `Buffer` classes.

The `AVPacket*` passed to `StreamProcessor` is first passed to `Decoder`.
`Decoder` generates audio / video frames (`AVFrame`) and pass it to `Sink`s.

Firstly `Sink` class passes the incoming frame to `FilterGraph`.

`FilterGraph` is a class based on [`AVFilterGraph` structure](https://ffmpeg.org/doxygen/4.1/structAVFilterGraph.html),
and it can apply various filters.
At minimum, it performs format conversion so that the resuling data is suitable for Tensor representation,
such as YUV to RGB.

The output `AVFrame` from `FilterGraph` is passed to `Buffer` class, which converts it to Tensor.

```
 StreamProcessor
┌─────────────────────────────────────────────────────────┐
│ AVPacket*                                               │
│  │                                                      │
│  │         AVFrame*          AVFrame*                   │
│  └► Decoder ──┬─► FilterGraph ─────► Buffer ───► Tensor │
│               │                                         │
│               ├─► FilterGraph ─────► Buffer ───► Tensor │
│               │                                         │
│               └─► ...                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Implementation guideline

### Memory management and object lifecycle

Ffmpeg uses raw pointers, which needs to be allocated and freed with dedicated functions.
In the binding code, these pointers are encapsulated in a class with RAII semantic and
`std::unique_ptr<>` to guarantee sole ownership.

**Decoder lifecycle**

```c++
// Default construction (no memory allocation)
decoder = Decoder(...);
// Decode
decoder.process_packet(pPacket);
// Retrieve result
decoder.get_frame(pFrame);
// Release resources
decoder::~Decoder();
```

**FilterGraph lifecycle**

```c++
// Default construction (no memory allocation)
filter_graph = FilterGraph(AVMEDIA_TYPE_AUDIO);
// Filter configuration
filter_fraph.add_audio_src(..)
filter_fraph.add_sink(..)
filter_fraph.add_process("<filter expression>")
filter_graph.create_filter();
// Apply filter
fitler_graph.add_frame(pFrame);
// Retrieve result
filter_graph.get_frame(pFrame);
// Release resources
filter_graph::~FilterGraph();
```

**StreamProcessor lifecycle**

```c++
// Default construction (no memory allocation)
processor = Processor(...);
// Define the process stream
processor.add_audio_stream(...);
processor.add_audio_stream(...);
// Process the packet
processor.process_packet(pPacket);
// Retrieve result
tensor = processor.get_chunk(...);
// Release resources
processor::~Processor();
```

### ON/OFF semantic and `std::unique_ptr<>`

Since we want to make some components (such as stream processors and filters)
separately configurable, we introduce states for ON/OFF.
To make the code simple, we use `std::unique_ptr<>`.
`nullptr` means the component is turned off.
This pattern applies to `StreamProcessor` (output streams).

### Exception and return value

To report the error during the configuration and initialization of objects,
we use `Exception`. However, throwing errors is expensive during the streaming,
so we use return value for that.
