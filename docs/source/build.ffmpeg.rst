.. _enabling_hw_decoder:

Enabling GPU video decoder/encoder
==================================

TorchAudio can make use of hardware-based video decoding and encoding supported by underlying FFmpeg libraries that are linked at runtime.

Using NVIDIA's GPU decoder and encoder, it is also possible to pass around CUDA Tensor directly, that is decode video into CUDA tensor or encode video from CUDA tensor, without moving data from/to CPU.

This improves the video throughput significantly. However, please note that not all the video formats are supported by hardware acceleration.

This page goes through how to build FFmpeg with hardware acceleration. For the detail on the performance of GPU decoder and encoder please see :ref:`NVDEC tutoial <nvdec_tutorial>` and :ref:`NVENC tutorial <nvenc_tutorial>`.

Overview
--------

Using them in TorchAduio requires additional FFmpeg configuration.

In the following, we look into how to enable GPU video decoding with `NVIDIA's Video codec SDK <https://developer.nvidia.com/nvidia-video-codec-sdk>`_.
To use NVENC/NVDEC with TorchAudio, the following items are required.

1. NVIDIA GPU with hardware video decoder/encoder.

2. FFmpeg libraries compiled with NVDEC/NVENC support. †

3. PyTorch / TorchAudio with CUDA support.

TorchAudio’s official binary distributions are compiled to work with FFmpeg libraries, and they contain the logic to use hardware decoding/encoding.

In the following, we build FFmpeg 4 libraries with NVDEC/NVENC support. You can also use FFmpeg 5 or 6.

The following procedure was tested on Ubuntu.

† For details on NVDEC/NVENC and FFmpeg, please refer to the following articles.

- https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/nvdec-video-decoder-api-prog-guide/
- https://docs.nvidia.com/video-technologies/video-codec-sdk/11.1/ffmpeg-with-nvidia-gpu/index.html#compiling-ffmpeg
- https://developer.nvidia.com/blog/nvidia-ffmpeg-transcoding-guide/

Check the GPU and CUDA version
------------------------------

First, check the available GPU. Here, we have Tesla T4 with CUDA Toolkit 11.2 installed.

.. code-block::

   $ nvidia-smi

   Fri Oct  7 13:01:26 2022
   +-----------------------------------------------------------------------------+
   | NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
   |-------------------------------+----------------------+----------------------+
   | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
   |                               |                      |               MIG M. |
   |===============================+======================+======================|
   |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
   | N/A   56C    P8    10W /  70W |      0MiB / 15109MiB |      0%      Default |
   |                               |                      |                  N/A |
   +-------------------------------+----------------------+----------------------+

   +-----------------------------------------------------------------------------+
   | Processes:                                                                  |
   |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
   |        ID   ID                                                   Usage      |
   |=============================================================================|
   |  No running processes found                                                 |
   +-----------------------------------------------------------------------------+

Checking the compute capability
-------------------------------

Later, we need the version of compute capability supported by this GPU. The following page lists the GPUs and corresponding compute capabilities. The compute capability of T4 is ``7.5``.

https://developer.nvidia.com/cuda-gpus

Install NVIDIA Video Codec Headers
----------------------------------

To build FFmpeg with NVDEC/NVENC, we first need to install the headers that FFmpeg uses to interact with Video Codec SDK.

Since we have CUDA 11 working in the system, we use one of ``n11`` tag.

.. code-block:: bash

   git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
   cd nv-codec-headers
   git checkout n11.0.10.1
   sudo make install

The location of installation can be changed with ``make PREFIX=<DESIRED_DIRECTORY> install``.
   
.. code-block:: text

   Cloning into 'nv-codec-headers'...
   remote: Enumerating objects: 819, done.
   remote: Counting objects: 100% (819/819), done.
   remote: Compressing objects: 100% (697/697), done.
   remote: Total 819 (delta 439), reused 0 (delta 0)
   Receiving objects: 100% (819/819), 156.42 KiB | 410.00 KiB/s, done.
   Resolving deltas: 100% (439/439), done.
   Note: checking out 'n11.0.10.1'.

   You are in 'detached HEAD' state. You can look around, make experimental
   changes and commit them, and you can discard any commits you make in this
   state without impacting any branches by performing another checkout.

   If you want to create a new branch to retain commits you create, you may
   do so (now or later) by using -b with the checkout command again. Example:

     git checkout -b <new-branch-name>

   HEAD is now at 315ad74 add cuMemcpy
   sed 's#@@PREFIX@@#/usr/local#' ffnvcodec.pc.in > ffnvcodec.pc
   install -m 0755 -d '/usr/local/include/ffnvcodec'
   install -m 0644 include/ffnvcodec/*.h '/usr/local/include/ffnvcodec'
   install -m 0755 -d '/usr/local/lib/pkgconfig'
   install -m 0644 ffnvcodec.pc '/usr/local/lib/pkgconfig'

Install FFmpeg dependencies
---------------------------

Next, we install tools and libraries required during the FFmpeg build.
The minimum requirement is `Yasm <https://yasm.tortall.net/>`_.
Here we additionally install H264 video codec and HTTPS protocol,
which we use later for verifying the installation.

.. code-block:: bash

   sudo apt -qq update
   sudo apt -qq install -y yasm libx264-dev libgnutls28-dev

.. code-block:: text

   ... Omitted for brevity ...

   STRIP   install-libavutil-shared
   Setting up libx264-dev:amd64 (2:0.152.2854+gite9a5903-2) ...
   Setting up yasm (1.3.0-2build1) ...
   Setting up libunbound2:amd64 (1.6.7-1ubuntu2.5) ...
   Setting up libp11-kit-dev:amd64 (0.23.9-2ubuntu0.1) ...
   Setting up libtasn1-6-dev:amd64 (4.13-2) ...
   Setting up libtasn1-doc (4.13-2) ...
   Setting up libgnutlsxx28:amd64 (3.5.18-1ubuntu1.6) ...
   Setting up libgnutls-dane0:amd64 (3.5.18-1ubuntu1.6) ...
   Setting up libgnutls-openssl27:amd64 (3.5.18-1ubuntu1.6) ...
   Setting up libgmpxx4ldbl:amd64 (2:6.1.2+dfsg-2) ...
   Setting up libidn2-dev:amd64 (2.0.4-1.1ubuntu0.2) ...
   Setting up libidn2-0-dev (2.0.4-1.1ubuntu0.2) ...
   Setting up libgmp-dev:amd64 (2:6.1.2+dfsg-2) ...
   Setting up nettle-dev:amd64 (3.4.1-0ubuntu0.18.04.1) ...
   Setting up libgnutls28-dev:amd64 (3.5.18-1ubuntu1.6) ...
   Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
   Processing triggers for libc-bin (2.27-3ubuntu1.6) ...

Build FFmpeg with NVDEC/NVENC support
-------------------------------------

Next we download the source code of FFmpeg 4. We use 4.4.2 here.

.. code-block:: bash

   wget -q https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.4.2.tar.gz
   tar -xf n4.4.2.tar.gz
   cd FFmpeg-n4.4.2

Next we configure FFmpeg build. Note the following:

1. We provide flags like ``-I/usr/local/cuda/include``, ``-L/usr/local/cuda/lib64`` to let the build process know where the CUDA libraries are found.
2. We provide flags like ``--enable-nvdec`` and ``--enable-nvenc`` to enable NVDEC/NVENC.
3. We also provide NVCC flags with compute capability ``75``, which corresponds to ``7.5`` of T4. †
4. We install the library in ``/usr/lib/``.

.. note::

   † The configuration script verifies NVCC by compiling a sample code. By default it uses old compute capability such as ``30``, which is no longer supported by CUDA 11. So it is required to set a correct compute capability.

.. code-block:: bash

   prefix=/usr/
   ccap=75

   ./configure \
     --prefix="${prefix}" \
     --extra-cflags='-I/usr/local/cuda/include' \
     --extra-ldflags='-L/usr/local/cuda/lib64' \
     --nvccflags="-gencode arch=compute_${ccap},code=sm_${ccap} -O2" \
     --disable-doc \
     --enable-decoder=aac \
     --enable-decoder=h264 \
     --enable-decoder=h264_cuvid \
     --enable-decoder=rawvideo \
     --enable-indev=lavfi \
     --enable-encoder=libx264 \
     --enable-encoder=h264_nvenc \
     --enable-demuxer=mov \
     --enable-muxer=mp4 \
     --enable-filter=scale \
     --enable-filter=testsrc2 \
     --enable-protocol=file \
     --enable-protocol=https \
     --enable-gnutls \
     --enable-shared \
     --enable-gpl \
     --enable-nonfree \
     --enable-cuda-nvcc \
     --enable-libx264 \
     --enable-nvenc \
     --enable-cuvid \
     --enable-nvdec   

.. code-block:: text

   install prefix            /usr/
   source path               .
   C compiler                gcc
   C library                 glibc
   ARCH                      x86 (generic)
   big-endian                no
   runtime cpu detection     yes
   standalone assembly       yes
   x86 assembler             yasm
   MMX enabled               yes
   MMXEXT enabled            yes
   3DNow! enabled            yes
   3DNow! extended enabled   yes
   SSE enabled               yes
   SSSE3 enabled             yes
   AESNI enabled             yes
   AVX enabled               yes
   AVX2 enabled              yes
   AVX-512 enabled           yes
   XOP enabled               yes
   FMA3 enabled              yes
   FMA4 enabled              yes
   i686 features enabled     yes
   CMOV is fast              yes
   EBX available             yes
   EBP available             yes
   debug symbols             yes
   strip symbols             yes
   optimize for size         no
   optimizations             yes
   static                    no
   shared                    yes
   postprocessing support    no
   network support           yes
   threading support         pthreads
   safe bitstream reader     yes
   texi2html enabled         no
   perl enabled              yes
   pod2man enabled           yes
   makeinfo enabled          no
   makeinfo supports HTML    no

   External libraries:
   alsa                    libx264                 lzma
   bzlib                   libxcb                  zlib
   gnutls                  libxcb_shape
   iconv                   libxcb_xfixes

   External libraries providing hardware acceleration:
   cuda                    cuvid                   nvenc
   cuda_llvm               ffnvcodec               v4l2_m2m
   cuda_nvcc               nvdec

   Libraries:
   avcodec                 avformat                swscale
   avdevice                avutil
   avfilter                swresample

   Programs:
   ffmpeg                  ffprobe

   Enabled decoders:
   aac                     hevc                    rawvideo
   av1                     mjpeg                   vc1
   h263                    mpeg1video              vp8
   h264                    mpeg2video              vp9
   h264_cuvid              mpeg4

   Enabled encoders:
   h264_nvenc              libx264

   Enabled hwaccels:
   av1_nvdec               mpeg1_nvdec             vp8_nvdec
   h264_nvdec              mpeg2_nvdec             vp9_nvdec
   hevc_nvdec              mpeg4_nvdec             wmv3_nvdec
   mjpeg_nvdec             vc1_nvdec

   Enabled parsers:
   h263                    mpeg4video              vp9

   Enabled demuxers:
   mov

   Enabled muxers:
   mov                     mp4

   Enabled protocols:
   file                    tcp
   https                   tls

   Enabled filters:
   aformat                 hflip                   transpose
   anull                   null                    trim
   atrim                   scale                   vflip
   format                  testsrc2

   Enabled bsfs:
   aac_adtstoasc           null                    vp9_superframe_split
   h264_mp4toannexb        vp9_superframe
   
   Enabled indevs:
   lavfi

   Enabled outdevs:

   License: nonfree and unredistributable

Now we build and install

.. code-block:: bash

   make clean
   make -j
   sudo make install

.. code-block:: text

   ... Omitted for brevity ...

   INSTALL libavdevice/libavdevice.so
   INSTALL libavfilter/libavfilter.so
   INSTALL libavformat/libavformat.so
   INSTALL libavcodec/libavcodec.so
   INSTALL libswresample/libswresample.so
   INSTALL libswscale/libswscale.so
   INSTALL libavutil/libavutil.so
   INSTALL install-progs-yes
   INSTALL ffmpeg
   INSTALL ffprobe

Checking the intallation
------------------------

To verify that the FFmpeg we built have CUDA support, we can check the list of available decoders and encoders.

.. code-block:: bash

   ffprobe -hide_banner -decoders | grep h264

.. code-block:: text

    VFS..D h264                 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10
    V..... h264_cuvid           Nvidia CUVID H264 decoder (codec h264)

.. code-block:: bash

   ffmpeg -hide_banner -encoders | grep 264

.. code-block:: text

    V..... libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
    V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)

The following command fetches video from remote server, decode with NVDEC (cuvid) and re-encode with NVENC. If this command does not work, then there is an issue with FFmpeg installation, and TorchAudio would not be able to use them either.

.. code-block:: bash

   $ src="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4"

   $ ffmpeg -hide_banner -y -vsync 0 \
        -hwaccel cuvid \
        -hwaccel_output_format cuda \
        -c:v h264_cuvid \
        -resize 360x240 \
        -i "${src}" \
        -c:a copy \
        -c:v h264_nvenc \
        -b:v 5M test.mp4

Note that there is ``Stream #0:0 -> #0:0 (h264 (h264_cuvid) -> h264 (h264_nvenc))``, which means that video is decoded with ``h264_cuvid`` decoder and ``h264_nvenc`` encoder.

.. code-block::

   Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4':
     Metadata:
       major_brand     : mp42
       minor_version   : 512
       compatible_brands: mp42iso2avc1mp41
       encoder         : Lavf58.76.100
     Duration: 00:03:26.04, start: 0.000000, bitrate: 1294 kb/s
     Stream #0:0(eng): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709), 960x540 [SAR 1:1 DAR 16:9], 1156 kb/s, 29.97 fps, 29.97 tbr, 30k tbn, 59.94 tbc (default)
       Metadata:
         handler_name    : ?Mainconcept Video Media Handler
         vendor_id       : [0][0][0][0]
     Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s (default)
       Metadata:
         handler_name    : #Mainconcept MP4 Sound Media Handler
         vendor_id       : [0][0][0][0]
   Stream mapping:
     Stream #0:0 -> #0:0 (h264 (h264_cuvid) -> h264 (h264_nvenc))
     Stream #0:1 -> #0:1 (copy)
   Press [q] to stop, [?] for help
   Output #0, mp4, to 'test.mp4':
     Metadata:
       major_brand     : mp42
       minor_version   : 512
       compatible_brands: mp42iso2avc1mp41
       encoder         : Lavf58.76.100
     Stream #0:0(eng): Video: h264 (Main) (avc1 / 0x31637661), cuda(tv, bt709, progressive), 360x240 [SAR 1:1 DAR 3:2], q=2-31, 5000 kb/s, 29.97 fps, 30k tbn (default)
       Metadata:
         handler_name    : ?Mainconcept Video Media Handler
         vendor_id       : [0][0][0][0]
         encoder         : Lavc58.134.100 h264_nvenc
       Side data:
         cpb: bitrate max/min/avg: 0/0/5000000 buffer size: 10000000 vbv_delay: N/A
     Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s (default)
       Metadata:
         handler_name    : #Mainconcept MP4 Sound Media Handler
         vendor_id       : [0][0][0][0]
   frame= 6175 fps=1712 q=11.0 Lsize=   37935kB time=00:03:26.01 bitrate=1508.5kbits/s speed=57.1x
   video:34502kB audio:3234kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.526932%

Using the GPU decoder/encoder from TorchAudio
---------------------------------------------

Checking the installation
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the FFmpeg is properly working with hardware acceleration, we need to check if TorchAudio can pick it up correctly.

There are utility functions to query the capability of FFmpeg in :py:mod:`torchaudio.utils.ffmpeg_utils`.

You can first use :py:func:`~torchaudio.utils.ffmpeg_utils.get_video_decoders` and :py:func:`~torchaudio.utils.ffmpeg_utils.get_video_encoders` to check if GPU decoders and encoders (such as ``h264_cuvid`` and ``h264_nvenc``) are listed.

It is often the case where there are multiple FFmpeg installations in the system, and TorchAudio is loading one different than expected. In such cases, use of ``ffmpeg`` to check the installation does not help. You can use functions like :py:func:`~torchaudio.utils.ffmpeg_utils.get_build_config` and :py:func:`~torchaudio.utils.ffmpeg_utils.get_versions` to get information about FFmpeg libraries TorchAudio loaded.

.. code-block:: python

   from torchaudio.utils import ffmpeg_utils

   print("Library versions:")
   print(ffmpeg_utils.get_versions())
   print("\nBuild config:")
   print(ffmpeg_utils.get_build_config())
   print("\nDecoders:")
   print([k for k in ffmpeg_utils.get_video_decoders().keys() if "cuvid" in k])
   print("\nEncoders:")
   print([k for k in ffmpeg_utils.get_video_encoders().keys() if "nvenc" in k])

.. code-block:: text

   Library versions:
   {'libavutil': (56, 31, 100), 'libavcodec': (58, 54, 100), 'libavformat': (58, 29, 100), 'libavfilter': (7, 57, 100), 'libavdevice': (58, 8, 100)}

   Build config:
   --prefix=/usr --extra-version=0ubuntu0.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-nvenc --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared

   Decoders:
   ['h264_cuvid', 'hevc_cuvid', 'mjpeg_cuvid', 'mpeg1_cuvid', 'mpeg2_cuvid', 'mpeg4_cuvid', 'vc1_cuvid', 'vp8_cuvid', 'vp9_cuvid']

   Encoders:
   ['h264_nvenc', 'nvenc', 'nvenc_h264', 'nvenc_hevc', 'hevc_nvenc']


Using the hardware decoder and encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the installation and the runtime linking work fine, then you can test the GPU decoding with the following.

For the detail on the performance of GPU decoder and encoder please see :ref:`NVDEC tutoial <nvdec_tutorial>` and :ref:`NVENC tutorial <nvenc_tutorial>`.
