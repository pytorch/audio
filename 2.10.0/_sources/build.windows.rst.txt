Building on Windows
===================

To build TorchAudio on Windows, we need to enable C++ compiler and install build tools and runtime dependencies.

We use Microsoft Visual C++ for compiling C++ and Conda for managing the other build tools and runtime dependencies.

1. Install build tools
----------------------

MSVC
~~~~

Please follow the instruction at https://visualstudio.microsoft.com/downloads/, and make sure to install C++ development tools.

.. note::

   The official binary distribution are compiled with MSVC 2022.
   The following section uses path from MSVC 2022 Community Edition.

Conda
~~~~~

Please follow the instruction at https://docs.conda.io/en/latest/miniconda.html.

2. Start the dev environment
----------------------------

In the following, we need to use C++ compiler (``cl``), and Conda package manager (``conda``).
We also use Bash for the sake of similar experience to Linux/macOS.

To do so, the following three steps are required.

1. Open command prompt
2. Enable developer environment
3. [Optional] Launch bash

|

The following combination is known to work.

1. Launch Anaconda3 Command Prompt.

   |

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/windows-conda.png
      :width: 360px

   |

   Please make sure that ``conda`` command is recognized.

   |

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/windows-conda2.png
      :width: 360px

   |

2. Activate dev tools by running the following command.

   We need to use the MSVC x64 toolset for compilation.
   To enable the toolset, one can use ``vcvarsall.bat`` or ``vcvars64.bat`` file, which
   are found under Visual Studio's installation folder, under ``VC\Auxiliary\Build\``.
   More information are available at https://docs.microsoft.com/en-us/cpp/build/how-to-enable-a-64-bit-visual-cpp-toolset-on-the-command-line?view=msvc-160#use-vcvarsallbat-to-set-a-64-bit-hosted-build-architecture

   .. code-block::

      call "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

   Please makes sure that ``cl`` command is recognized.

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/windows-msvc.png
      :width: 360px

3. [Optional] Launch bash with the following command.

   If you want a similar UX as Linux/macOS, you can launch Bash. However, please note that in Bash environment, the file paths are different from native Windows style, and ``torchaudio.datasets`` module does not work.

   .. code-block::

      Miniconda3\Library\bin\bash.exe

   .. image:: https://download.pytorch.org/torchaudio/doc-assets/windows-bash.png
      :width: 360px

3. Install PyTorch
------------------
Please refer to https://pytorch.org/get-started/locally/ for the up-to-date way to install PyTorch.

4. [Optional] cuDNN
-------------------

If you intend to build CUDA-related features, please install cuDNN.

Download CuDNN from https://developer.nvidia.com/cudnn, and extract files in
the same directories as CUDA toolkit.

When using conda, the directories are ``${CONDA_PREFIX}/bin``, ``${CONDA_PREFIX}/include``, ``${CONDA_PREFIX}/Lib/x64``.

5. Install external dependencies
--------------------------------

.. code-block::

   conda install cmake ninja

6. Build TorchAudio
-------------------

Now that we have everything ready, we can build TorchAudio.

.. code-block::

   git clone https://github.com/pytorch/audio
   cd audio


.. code-block::

   # In Command Prompt
   pip install -e . -v --no-build-isolation

.. code-block::

   # In Bash
   pip install -e . -v --no-build-isolation

.. note::
   Due to the complexity of build process, TorchAudio only supports in-place build.
   To use ``pip``, please use ``--no-use-pep517`` option.

   ``pip install -v -e . --no-use-pep517``
