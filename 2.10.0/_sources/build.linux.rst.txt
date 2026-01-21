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

   pip install --pre --index-url https://download.pytorch.org/whl/nightly/cpu

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

   pip install -e . -v --no-build-isolation

.. note::
   Due to the complexity of build process, TorchAudio only supports in-place build.
   To use ``pip``, please use ``--no-use-pep517`` option.

   ``pip install -v -e . --no-use-pep517``
