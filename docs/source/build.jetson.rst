Building on Jetson
==================

1. Install JetPack
------------------

JetPack includes the collection of CUDA-related libraries that is required to run PyTorch with CUDA.

Please refer to https://developer.nvidia.com/embedded/learn/get-started-jetson-agx-orin-devkit for the up-to-date instruction.

.. code-block::

   sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/common r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'
   sudo bash -c 'echo "deb https://repo.download.nvidia.com/jetson/t234 r34.1 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list'

   sudo apt update
   sudo apt dist-upgrade

   # REBOOT

   sudo apt install nvidia-jetpack

Checking the versions
~~~~~~~~~~~~~~~~~~~~~

To check the version installed you can use the following commands;

.. code-block::

   # JetPack
   $ apt list --installed | grep nvidia-jetpack

   nvidia-jetpack-dev/stable,now 5.0.1-b118 arm64 [installed,automatic]
   nvidia-jetpack-runtime/stable,now 5.0.1-b118 arm64 [installed,automatic]
   nvidia-jetpack/stable,now 5.0.1-b118 arm64 [installed]

   # CUDA
   $ apt list --installed | grep cuda-toolkit

   cuda-toolkit-11-4-config-common/stable,now 11.4.243-1 all [installed,automatic]
   cuda-toolkit-11-4/stable,now 11.4.14-1 arm64 [installed,automatic]
   cuda-toolkit-11-config-common/stable,now 11.4.243-1 all [installed,automatic]
   cuda-toolkit-config-common/stable,now 11.4.243-1 all [installed,automatic]

   # cuDNN
   $ apt list --installed | grep cudnn

   libcudnn8-dev/stable,now 8.3.2.49-1+cuda11.4 arm64 [installed,automatic]
   libcudnn8-samples/stable,now 8.3.2.49-1+cuda11.4 arm64 [installed,automatic]
   libcudnn8/stable,now 8.3.2.49-1+cuda11.4 arm64 [installed,automatic]
   nvidia-cudnn8-dev/stable,now 5.0.1-b118 arm64 [installed,automatic]
   nvidia-cudnn8-runtime/stable,now 5.0.1-b118 arm64 [installed,automatic]

.. image:: https://download.pytorch.org/torchaudio/doc-assets/jetson-package-versions.png
   :width: 360px

2. [Optional] Install jtop
--------------------------

Since Tegra GPUs are not supported by ``nvidia-smi`` command, it is recommended to isntall ``jtop``.

Only super-use can install ``jtop``. So make sure to add ``-U``, so that running ``jtop`` won't require super-user priviledge.

3. Install ``pip`` in user env
------------------------------

By default, ``pip`` / ``pip3`` commands use the ones from system directory ``/usr/bin/``, and its ``site-packages`` directory is protected and cannot be modified without ``sudo``.

One way to workaround this is to install ``pip`` in user directory.

https://forums.developer.nvidia.com/t/python-3-module-install-folder/181321

.. code-block::

   wget https://bootstrap.pypa.io/get-pip.py
   python get-pip.py --user

After this verify that ``pip`` command is pointing the one in user directory.

.. code-block::

   $ which pip
   /home/USER/.local/bin/pip

4. Install PyTorch
------------------

As of PyTorch 1.13 and torchaudio 0.13, there is no official pre-built binaries for Linux ARM64. Nidia provides custom pre-built binaries for PyTorch, which works with specific JetPack.

Please refer to https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html for up-to-date instruction on how to install PyTorch.

.. code-block::

   $ package=torch-1.13.0a0+340c4120.nv22.06-cp38-cp38-linux_aarch64.whl
   $ wget "https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/${package}"
   $ pip install --no-cache "${package}"

Verify the installation by checking the version and CUDA device accessibility.

.. code-block::

   $ python -c '

   import torch
   print(torch.__version__)
   print(torch.cuda.is_available())
   print(torch.empty((1, 2), device=torch.device("cuda")))
   '
   1.13.0a0+410ce96a.nv22.12
   True
   tensor([[0., 0.]], device='cuda:0')

.. image:: https://download.pytorch.org/torchaudio/doc-assets/jetson-torch.png
   :width: 360px

5. Build TorchAudio
-------------------

1. Install build tools
~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   pip install cmake ninja

2. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   sudo apt install ffmpeg libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev libavfilter-dev

3. Build TorchAudio
~~~~~~~~~~~~~~~~~~~

.. code-block::

   git clone https://github.com/pytorch/audio
   cd audio
   USE_CUDA=1 pip install -v -e . --no-use-pep517

4. Check the installation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

   import torchaudio

   print(torchaudio.__version__)

   torchaudio.utils.ffmpeg_utils.get_build_config()

.. code-block::

   2.0.0a0+2ead941
   --prefix=/usr --extra-version=0ubuntu0.1 --toolchain=hardened --libdir=/usr/lib/aarch64-linux-gnu --incdir=/usr/include/aarch64-linux-gnu --arch=arm64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared

.. image:: https://download.pytorch.org/torchaudio/doc-assets/jetson-verify-build.png
   :width: 360px
