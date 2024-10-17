Building on Linux and macOS
===========================

1. Install Conda and activate conda environment
-----------------------------------------------

Please folllow the instruction at https://docs.conda.io/en/latest/miniconda.html

2. Install PyTorch
------------------

Please select the version of PyTorch you want to install from https://pytorch.org/get-started/locally/

Here, we install nightly build.

.. code-block::

   conda install pytorch -c pytorch-nightly

3. Install build tools
----------------------

.. code-block::

   conda install cmake ninja

4. Clone the torchaudio repository
----------------------------------

.. code-block::

   git clone https://github.com/pytorch/audio
   cd audio

5. Build
--------

.. code-block::

   python setup.py develop

.. note::
   Due to the complexity of build process, TorchAudio only supports in-place build.
   To use ``pip``, please use ``--no-use-pep517`` option.

   ``pip install -v -e . --no-use-pep517``

[Optional] Build TorchAudio with a custom built FFmpeg
------------------------------------------------------

By default, torchaudio tries to build FFmpeg extension with support for multiple FFmpeg versions. This process uses pre-built FFmpeg libraries compiled for specific CPU architectures like ``x86_64`` and ``aarch64`` (``arm64``).

If your CPU is not one of those, then the build process can fail. To workaround, one can disable FFmpeg integration (by setting the environment variable ``USE_FFMPEG=0``) or switch to the single version FFmpeg extension.

To build single version FFmpeg extension, FFmpeg binaries must be provided by user and available in the build environment. To do so, install FFmpeg and set ``FFMPEG_ROOT`` environment variable to specify the location of FFmpeg.

.. code-block::

   conda install -c conda-forge ffmpeg
   FFMPEG_ROOT=${CONDA_PREFIX} python setup.py develop
