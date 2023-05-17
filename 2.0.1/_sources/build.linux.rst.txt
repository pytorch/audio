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

   conda install cmake ninja pkg-config

4. Install external dependencies
--------------------------------

.. code-block::

   conda install -c conda-forge ffmpeg

5. Clone the torchaudio repository
----------------------------------

.. code-block::

   git clone https://github.com/pytorch/audio
   cd audio

6. Build
--------

.. code-block::

   USE_FFMPEG=1 python setup.py develop

.. note::
   Due to the complexity of build process, TorchAudio only supports in-place build.
   To use ``pip``, please use ``--no-use-pep517`` option.

   ``USE_FFMPEG=1 pip install -v -e . --no-use-pep517``
