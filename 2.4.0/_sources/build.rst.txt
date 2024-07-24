Building from source
====================

TorchAudio integrates PyTorch for numerical computation and third party libraries for multimedia I/O. It requires the following tools to build from source.

- `PyTorch <https://pytorch.org>`_
- `CMake <https://cmake.org/>`_
- `Ninja <https://ninja-build.org/>`_
- C++ complier with C++ 17 support
   - `GCC <https://gcc.gnu.org/>`_ (Linux)
   - `Clang <https://clang.llvm.org/>`_ (macOS)
   - `MSVC <https://visualstudio.microsoft.com>`_  2019 or newer (Windows)
- `CUDA toolkit <https://developer.nvidia.com/cudnn>`_ and `cuDNN <https://developer.nvidia.com/cudnn>`_ (if building CUDA extension)

Most of the tools are available in `Conda <https://conda.io/>`_, so we recommend using conda.

.. toctree::
   :maxdepth: 1

   build.linux
   build.windows
   build.jetson

Customizing the build
---------------------

TorchAudio's integration with third party libraries can be enabled/disabled via
environment variables.

They can be enabled by passing ``1`` and disabled by ``0``.

- ``BUILD_SOX``: Enable/disable I/O features based on libsox.
- ``BUILD_KALDI``: Enable/disable feature extraction based on Kaldi.
- ``BUILD_RNNT``: Enable/disable custom RNN-T loss function.
- ``USE_FFMPEG``: Enable/disable I/O features based on FFmpeg libraries.
- ``USE_ROCM``: Enable/disable AMD ROCm support.
- ``USE_CUDA``: Enable/disable CUDA support.

For the latest configurations and their default values, please check the source code.
https://github.com/pytorch/audio/blob/main/tools/setup_helpers/extension.py
